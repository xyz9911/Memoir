#!/usr/bin/env python3
"""Sequential frozen-cache -> complete WM pretraining -> complete IR2R queue.

Run under tmux. Each stage succeeds only when its verified completion marker
exists. A failed training stage is never silently restarted from initialization.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from drama_pretrain import (sha, json_sha, write_json, space_guard, verify_host_volume,
                            load_selected_weights, validate_cache_contract, verify_storage_mount, code_manifest)
from drama_pretrain import configure_runtime

def validate_completed_stage(name,cfg,expected_marker_sha=None):
    marker=(Path(cfg['cache']).with_suffix('.manifest.json') if name=='cache'
            else Path(cfg['output_path'] if name=='evaluation' else cfg['run_dir'])/'completion.json')
    if expected_marker_sha is not None and sha(marker)!=expected_marker_sha:
        raise RuntimeError(f'{name} completion marker changed')
    outcome=json.loads(marker.read_text())
    if not outcome.get('complete'):raise RuntimeError(f'{name} incomplete; refusing queue advancement')
    if name=='evaluation':
        from drama_evaluate import validate_results
        validate_results(cfg)
    elif name=='cache':
        manifest=validate_cache_contract(cfg)
        if sha(cfg['cache'])!=manifest['sha256']:raise RuntimeError('Sealed cache payload checksum failed')
    else:
        step=outcome['step'] if name=='pretrain' else outcome['actual_updates']
        target=cfg['steps'] if name=='pretrain' else cfg['minimum_updates']
        if step<target or (name=='pretrain' and step!=target):raise RuntimeError('Training update budget incomplete')
        local,backup=Path(cfg['run_dir']),Path(cfg['backup_dir'])
        identity=json.loads((local/'identity.json').read_text())
        if json.loads((backup/'identity.json').read_text())!=identity:raise RuntimeError('Backup identity differs')
        if identity['contract']['config']!=cfg:raise RuntimeError('Completed stage config differs')
        if identity['contract']['source_files']!=code_manifest():raise RuntimeError('Completed stage source differs')
        if sha(backup/'frozen-base.pt')!=identity['frozen_base_sha256']:raise RuntimeError('Frozen base backup is corrupt')
        refs=json.loads((backup/'checkpoints'/'refs.json').read_text())
        for which in ('latest','best'):
            pointer=refs.get(which)
            if not pointer:raise RuntimeError(f'Completed {name} missing {which} checkpoint')
            directory=backup/'checkpoints'/'snapshots'/pointer['checkpoint_id']
            digest=sha(directory/'manifest.json');manifest=json.loads((directory/'manifest.json').read_text())
            if (digest!=pointer['manifest_sha256'] or (directory/'COMMITTED').read_text().strip()!=digest
                    or manifest['run_id']!=refs['run_id'] or pointer['run_id']!=refs['run_id']
                    or (which=='best' and not manifest['is_best'])):raise RuntimeError('Checkpoint reference or commit differs')
            for filename,record in manifest['files'].items():
                if sha(directory/filename)!=record['sha256']:raise RuntimeError('Checkpoint backup is corrupt')
            if (which=='latest' and manifest['step']!=step) or manifest['step']>step:
                raise RuntimeError('Checkpoint step disagrees with completion marker')
    return sha(marker)

def run(pretrain_config,ir2r_config,eval_config,journal_path,resume=False):
    # These are inherited by newly launched Python interpreters, before CUDA/HF init.
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED='0',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    configure_runtime()
    pre_path,ir_path,eval_path,journal_path=map(Path,(pretrain_config,ir2r_config,eval_config,journal_path))
    pre=json.loads(pre_path.read_text());ir=json.loads(ir_path.read_text());evaluation=json.loads(eval_path.read_text())
    from drama_evaluate import storage_config
    verify_storage_mount(pre);verify_storage_mount(ir);verify_storage_mount(storage_config(evaluation))
    if evaluation['ir2r_run']!=ir['run_dir'] or evaluation['ir2r_backup']!=ir['backup_dir']:
        raise RuntimeError('Final evaluation must compare this pipeline IR2R experiment')
    if not journal_path.resolve().is_relative_to(Path(pre['storage_mount']).resolve()):
        raise RuntimeError('Pipeline journal must be on the selected storage mount')
    contract={'pretrain_config_sha256':sha(pre_path),'ir2r_config_sha256':sha(ir_path),
              'evaluation_config_sha256':sha(eval_path),'pipeline_sha256':sha(__file__)}
    verify_host_volume(pre['host_volume'])
    if journal_path.exists():
        if not resume:raise RuntimeError('Existing pipeline requires --require-resume')
        journal=json.loads(journal_path.read_text())
        if journal['contract']!=contract:raise RuntimeError('Pipeline config/source identity changed')
    else:
        if resume:raise RuntimeError('No pipeline journal; strict resume refused')
        journal_path.parent.mkdir(parents=True,exist_ok=True)
        journal={'schema':'memoir-drama-pipeline-v1','contract':contract,'stages':{},'complete':False}
        write_json(journal_path,journal)
    scripts=Path(__file__).resolve().parent
    for name,command,cfg in (
        ('cache',[sys.executable,'-u',str(scripts/'drama_pretrain.py'),'cache','--config',str(pre_path)],pre),
        ('pretrain',[sys.executable,'-u',str(scripts/'drama_pretrain.py'),'train','--config',str(pre_path)],pre),
        ('ir2r',[sys.executable,'-u',str(scripts/'drama_ir2r.py'),'--config',str(ir_path)],ir),
        ('evaluation',[sys.executable,'-u',str(scripts/'drama_evaluate.py'),'--config',str(eval_path)],evaluation)):
        prior=journal['stages'].get(name)
        verify_storage_mount(storage_config(cfg) if name=='evaluation' else cfg)
        if prior and prior.get('complete'):
            validate_completed_stage(name,cfg,prior['completion_sha256'])
            continue
        verify_host_volume(cfg['host_volume'])
        space_guard([cfg['host_volume']],cfg['min_free_gib'])
        if name!='cache':
            for key in (('output_path','backup_path') if name=='evaluation' else ('run_dir','backup_dir')):
                Path(cfg[key]).parent.mkdir(parents=True,exist_ok=True)
            if prior:command.append('--require-resume')
        if name=='ir2r':
            done=json.loads((Path(pre['run_dir'])/'completion.json').read_text())
            if not done['complete'] or done['step']!=pre['steps']:raise RuntimeError('Pretraining did not complete configured updates')
            weights,metadata=load_selected_weights(pre['run_dir'],pre['backup_dir']);del weights
            journal['pretraining_selected_step']=metadata['selected_step']
        journal['stages'][name]={'complete':False,'started_unix':time.time(),'command':command}
        write_json(journal_path,journal)
        print('PIPELINE_STAGE',name,flush=True)
        result=subprocess.run(command)
        if result.returncode:
            journal['stages'][name]['exit_code']=result.returncode;write_json(journal_path,journal)
            raise RuntimeError(f'{name} failed with exit code {result.returncode}; inspect artifacts then strict resume')
        completion_sha=validate_completed_stage(name,cfg)
        journal['stages'][name].update(complete=True,finished_unix=time.time(),completion_sha256=completion_sha)
        write_json(journal_path,journal)
    journal['complete']=True;journal['finished_unix']=time.time();write_json(journal_path,journal)
    print('PIPELINE_COMPLETE',str(journal_path),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pretrain-config',required=True);p.add_argument('--ir2r-config',required=True)
    p.add_argument('--eval-config',required=True)
    p.add_argument('--journal',required=True);p.add_argument('--require-resume',action='store_true');a=p.parse_args()
    run(a.pretrain_config,a.ir2r_config,a.eval_config,a.journal,a.require_resume)

if __name__=='__main__':main()
