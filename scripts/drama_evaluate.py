#!/usr/bin/env python3
"""Complete train_dev author-tour comparison against the released TSSM reference.

Each tour has its own fixed RNG seed and starts with empty tour memory. Memory
persists across all instructions within that tour. Verified completed tour
artifacts survive interruption; an incomplete tour is never reported as done.
This is a released-reference comparison, not a matched-budget ablation or test.
"""
import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
from drama_pretrain import (sha,json_sha,write_json,verified_copy,code_manifest,
    runtime_identity,seed_all,space_guard,verify_storage_mount,verify_host_volume)
from drama_pretrain import configure_runtime
from drama_ir2r import (SCHEMA as IR2R_SCHEMA,author_args,select_split,require)

SCHEMA='memoir-drama-train-dev-comparison-v1'
PHASES=('released_tssm','selected_drama')
LIMITATION=('Released TSSM reference was trained on the original full training set and '
    'selected by its authors using val_unseen. Drama fits train_fit and selects on train_dev. '
    'Both initializations can inherit original training-data exposure. This train_dev '
    'comparison is neither a matched-data/budget ablation nor a blind generalization test. '
    'No val_unseen annotations or trajectories are loaded by this evaluator.')


def load_ir2r_selection(cfg,load_weights=True):
    """Read authoritative best IR2R compact state; never use the pretrain loader."""
    run,backup=Path(cfg['ir2r_run']),Path(cfg['ir2r_backup'])
    complete=json.loads((run/'completion.json').read_text())
    identity=json.loads((run/'identity.json').read_text())
    require(identity==json.loads((backup/'identity.json').read_text()),'IR2R backup identity differs')
    require(identity['contract']['schema']==IR2R_SCHEMA,'Unexpected IR2R schema')
    target=identity['contract']['config']['minimum_updates']
    require(target>=20000 and complete.get('complete') is True and complete['actual_updates']>=target,
            'Final comparison requires a complete production IR2R run')
    refs=json.loads((backup/'checkpoints'/'refs.json').read_text())
    payloads={}
    for selection in ('best','latest'):
        pointer=refs[selection];require(bool(pointer),'IR2R selection is absent')
        directory=backup/'checkpoints'/'snapshots'/pointer['checkpoint_id']
        manifest=json.loads((directory/'manifest.json').read_text());digest=sha(directory/'manifest.json')
        require(digest==pointer['manifest_sha256']==(directory/'COMMITTED').read_text().strip(),
                'IR2R checkpoint commit differs')
        require(manifest['run_id']==pointer['run_id']==refs['run_id'],'IR2R run ID differs')
        require(manifest['checkpoint_id']==pointer['checkpoint_id'],'IR2R checkpoint ID differs')
        require(set(manifest['files'])=={'state.pt','head.pt'},'Unexpected IR2R snapshot files')
        require(selection!='best' or manifest['is_best'],'Best reference does not name a best checkpoint')
        for name,record in manifest['files'].items():
            require(sha(directory/name)==record['sha256'],'IR2R checkpoint checksum differs')
        expected=complete['best_step'] if selection=='best' else complete['actual_updates']
        require(manifest['step']==expected,'IR2R selected/latest step disagrees with completion')
        payloads[selection]=(directory,manifest,digest)
    require(sha(backup/'frozen-base.pt')==identity['frozen_base_sha256'],'IR2R frozen base differs')
    directory,manifest,digest=payloads['best']
    metadata={'identity':identity,'completion':complete,'selected_manifest':manifest,
              'selected_manifest_sha256':digest,'selected_step':manifest['step']}
    if not load_weights:return None,metadata
    state=torch.load(directory/'state.pt',map_location='cpu',weights_only=True)
    require(state['schema']==IR2R_SCHEMA and state['identity_sha256']==json_sha(identity),'IR2R state identity differs')
    require(state['step']==manifest['step']==state['progress']['completed_updates'],'IR2R state step differs')
    require(state['progress']['best_step']==state['step'],'Selected snapshot is not the recorded best')
    frozen=torch.load(backup/'frozen-base.pt',map_location='cpu',weights_only=True)
    delta=state['trainable_state']
    require(set(delta)==set(identity['contract']['delta_names']),'IR2R changing-state keys differ')
    require(not set(frozen)&set(delta),'IR2R frozen/delta overlap')
    frozen.update(delta)
    return frozen,metadata


def tour_seed(base,tour_id):
    return (int(base)+int(json_sha(tour_id)[:8],16))%(2**32)


def protocol(cfg,training):
    from nav_src.envs.r2r.data_utils import construct_instrs
    split=json.loads(Path(training['split_manifest']).read_text())
    annotations=Path(training['datasets'])/'R2R'/'annotations'/'R2R_train_enc.json'
    items=construct_instrs(str(annotations.parent),'r2r',['train'],'bert',512)
    tours=json.loads(Path(training['tours']).read_text())['train']
    selected,subset=select_split(items,tours,split['train_dev'])
    require(len(split['train_dev'])==12 and len(selected)==2889,'Final comparison must cover all frozen train_dev')
    by_id={x['instr_id']:x for x in selected};records=[]
    for scan in sorted(subset):
        for index,members in enumerate(subset[scan]):
            key=f'{scan}__{index:04d}'
            records.append({'tour_id':key,'scan':scan,'members':members,
                            'seed':tour_seed(cfg['evaluation_seed'],key)})
    require(len(records)==36,'Expected all 36 train_dev author tours')
    flattened=[i for r in records for i in r['members']]
    require(len(flattened)==len(set(flattened))==2889,'Official tours duplicate or omit dev instructions')
    return by_id,records


def score_rows(env,predictions,expected_ids):
    require(len(predictions)==len(expected_ids),'Tour prediction count differs')
    require({x['instr_id'] for x in predictions}==set(expected_ids),'Tour prediction coverage differs')
    rows=[]
    for item in predictions:
        key=item['instr_id'];scan,gt=env.gt_trajs[key]
        scores=env._eval_item(scan,item['path'],gt)
        values={k:float(v) for k,v in scores.items()}
        require(all(math.isfinite(v) for v in values.values()),'Nonfinite trajectory metric')
        rows.append({'instr_id':key,'scan':scan,'trajectory':item['path'],'gt_path':gt,'scores':values})
    return rows


def evaluate_tour(agent,env,record):
    agent.env=env;agent.gmaps=[];agent.scanvp_cands={};agent.store_obs={}
    agent.logs=defaultdict(list);agent.results={};agent.full_results=[]
    agent.feedback='argmax';agent.loss=0;agent.losses=[];agent.vln_bert.eval()
    env.reset_epoch(shuffle=False);seed_all(record['seed'])
    predictions=[]
    with torch.no_grad():
        # Exactly one full tour; avoid BaseAgent.test's repeated sentinel tour.
        for _ in range(len(record['members'])):
            predictions.extend(agent.rollout(extended_memory=True))
            if env.check_last():break
    require(env.check_last(),'Partial tour evaluation rejected')
    return score_rows(env,predictions,record['members'])


def aggregate(rows):
    require(bool(rows),'Cannot summarize empty evaluation')
    require(len({x['instr_id'] for x in rows})==len(rows),'Duplicate final evaluation IDs')
    result={'instructions':len(rows)}
    for output,key,scale in [('SR','success',100.),('SPL','spl',100.),('NE','nav_error',1.),('nDTW','nDTW',100.)]:
        result[output]=float(np.mean([x['scores'][key] for x in rows]))*scale
    return result


def comparison(baseline,drama):
    before={x['instr_id']:x for x in baseline};after={x['instr_id']:x for x in drama}
    require(set(before)==set(after),'Models evaluated different instruction sets')
    a,b=aggregate(baseline),aggregate(drama)
    return {'released_tssm':a,'selected_drama':b,
        'delta_drama_minus_released':{k:b[k]-a[k] for k in ('SR','SPL','NE','nDTW')},
        'delta_units':{'SR':'percentage points','SPL':'percentage points','NE':'meters','nDTW':'percentage points'},
        'goal_sr_plus5_spl_nondecreasing':b['SR']-a['SR']>=5 and b['SPL']>=a['SPL'],
        'paired':[{'instr_id':key,'scan':before[key]['scan'],
            'released_tssm':before[key]['scores'],'selected_drama':after[key]['scores'],
            'delta':{k:after[key]['scores'][k]-before[key]['scores'][k] for k in ('success','spl','nav_error','nDTW')}}
            for key in sorted(before)]}


def artifact_paths(root,phase,tour_id):
    directory=Path(root)/'tours'/phase
    return directory/(tour_id+'.json'),directory/(tour_id+'.sha256')


def read_tour(root,phase,record,identity_sha,allow_uncommitted=False):
    path,marker=artifact_paths(root,phase,record['tour_id'])
    if not path.exists() and not marker.exists():return None
    if allow_uncommitted and path.is_file() and not marker.exists():return None
    require(path.is_file() and marker.is_file(),'Incomplete tour artifact/commit pair; refuse silent restart')
    require(sha(path)==marker.read_text().strip(),'Tour artifact checksum differs')
    data=json.loads(path.read_text())
    require(data['identity_sha256']==identity_sha and data['phase']==phase and data['tour']==record,
            'Tour artifact experiment identity differs')
    require(data.get('complete') is True,'Incomplete tour artifact')
    rows=data['rows'];require(len(rows)==len(record['members']) and
        {r['instr_id'] for r in rows}==set(record['members']),'Committed tour coverage differs')
    require(all(math.isfinite(float(v)) for r in rows for v in r['scores'].values()),'Nonfinite committed metric')
    return data


def publish_tour(output,backup,phase,record,identity_sha,data):
    """Only an uncommitted own result may be replaced after interruption."""
    from drama_checkpoint_store import _atomic_bytes
    local,local_commit=artifact_paths(output,phase,record['tour_id'])
    remote,remote_commit=artifact_paths(backup,phase,record['tour_id'])
    require(not remote_commit.exists(),'Refusing to overwrite a committed tour')
    require(not local_commit.exists(),'Committed local tour exists without backup; refuse overwrite')
    local.parent.mkdir(parents=True,exist_ok=True);remote.parent.mkdir(parents=True,exist_ok=True)
    write_json(local,data);digest=sha(local)
    _atomic_bytes(remote,local.read_bytes())
    require(sha(remote)==digest,'Tour backup readback failed')
    # Backup is authoritative: a crash before this marker redoes only this tour.
    _atomic_bytes(remote_commit,(digest+'\n').encode())
    _atomic_bytes(local_commit,(digest+'\n').encode())
    require(read_tour(backup,phase,record,identity_sha) is not None,'Backup tour verification failed')


def restore_local_tour(output,backup,phase,record,identity_sha):
    """Restore a missing/interrupted local copy from its verified backup."""
    data=read_tour(backup,phase,record,identity_sha)
    require(data is not None,'No verified backup tour')
    local,local_commit=artifact_paths(output,phase,record['tour_id'])
    remote,remote_commit=artifact_paths(backup,phase,record['tour_id'])
    from drama_checkpoint_store import _atomic_bytes
    local.parent.mkdir(parents=True,exist_ok=True)
    if not local.is_file() or sha(local)!=sha(remote):_atomic_bytes(local,remote.read_bytes())
    _atomic_bytes(local_commit,remote_commit.read_bytes())
    require(read_tour(output,phase,record,identity_sha)==data,'Local tour recovery failed')


def storage_config(cfg):
    return {**cfg,'run_dir':cfg['output_path'],'backup_dir':cfg['backup_path']}


def validate_results(cfg):
    run,backup=Path(cfg['output_path']),Path(cfg['backup_path'])
    identity=json.loads((run/'identity.json').read_text())
    require(identity==json.loads((backup/'identity.json').read_text()),'Evaluation identities differ')
    require(identity['config']==cfg and identity['source_files']==code_manifest(),'Evaluation config/source differs')
    marker=json.loads((run/'completion.json').read_text())
    require(marker.get('complete') is True,'Comparison incomplete')
    all_rows={}
    for phase in PHASES:
        rows=[]
        for record in identity['tours']:
            artifact=read_tour(backup,phase,record,json_sha(identity))
            require(artifact is not None,'Completed comparison missing tour')
            require(read_tour(run,phase,record,json_sha(identity))==artifact,'Local/backup tour artifacts differ')
            rows.extend(artifact['rows'])
        require(len(rows)==2889,'Incomplete production comparison')
        all_rows[phase]=rows
    expected=comparison(all_rows[PHASES[0]],all_rows[PHASES[1]])
    for root in (run,backup):
        require(sha(root/'comparison.json')==marker['comparison_sha256'],'Comparison checksum differs')
        require(json.loads((root/'comparison.json').read_text())==expected,'Comparison does not match trajectory artifacts')
    return marker


def run(cfg,resume=False):
    configure_runtime(cfg)
    from nav_src.envs.r2r.iter_env import IR2RNavBatch
    from nav_src.iter_nav.agent_ir2r import GMapNavAgent
    from utils.data import ImageFeaturesDB
    require(cfg.get('schema')==SCHEMA and cfg.get('scope')=='complete_train_dev_author_tours','Production evaluation scope required')
    require(cfg.get('backup_backend')=='filesystem','Explicit filesystem backup required')
    require(os.environ.get('PYTHONHASHSEED')=='0','PYTHONHASHSEED=0 required')
    storage=storage_config(cfg);mount=verify_storage_mount(storage)
    host=verify_host_volume(cfg['host_volume'])
    output,backup=Path(cfg['output_path']),Path(cfg['backup_path'])
    require(output.resolve()!=backup.resolve() and not output.resolve().is_relative_to(backup.resolve()) and
        not backup.resolve().is_relative_to(output.resolve()),'Evaluation backup must be separate')
    def guard():
        verify_storage_mount(storage,mount);verify_host_volume(cfg['host_volume'],host)
        space_guard([cfg['storage_mount'],cfg['host_volume']],cfg['min_free_gib'])
    guard()
    require(sha(cfg['released_checkpoint'])==cfg['released_checkpoint_sha256'],'Released reference checkpoint SHA differs')
    _,selection=load_ir2r_selection(cfg,load_weights=False)
    training=selection['identity']['contract']['config']
    require(selection['identity']['contract']['source_files']==code_manifest(),'Completed training source differs')
    assets=selection['identity']['contract']['assets']
    for path,digest in assets.items():require(sha(path)==digest,'Evaluation input asset differs: '+path)
    by_id,tours=protocol(cfg,training)
    identity={'schema':SCHEMA,'config':cfg,'source_files':code_manifest(),'runtime':runtime_identity(),
        'selection':selection,'assets':assets,'tours':tours,'limitation':LIMITATION,
        'inference':'batch1, argmax, dropout off, extended memory within full author tour; fresh memory and deterministic seed per tour',
        'storage_mount':mount,'host_mount':host}
    identity_sha=json_sha(identity)
    if resume:
        require(json.loads((output/'identity.json').read_text())==identity and
                json.loads((backup/'identity.json').read_text())==identity,'Strict comparison resume identity differs')
    else:
        require(not output.exists() and not backup.exists(),'Fresh comparison refuses existing directories')
        output.mkdir(parents=True);backup.mkdir(parents=True)
        write_json(output/'identity.json',identity);verified_copy(output/'identity.json',backup/'identity.json')
    for phase in PHASES:
        agent=None
        for record in tours:
            guard()
            existing=read_tour(backup,phase,record,identity_sha,allow_uncommitted=True)
            if existing is not None:
                restore_local_tour(output,backup,phase,record,identity_sha)
                continue
            _,local_commit=artifact_paths(output,phase,record['tour_id'])
            require(not local_commit.exists(),'Committed local tour exists without backup; restore verified backup before resuming')
            args_cfg={**training,'batch_size':1,'run_dir':str(output/'author-layout'/phase)}
            architecture=selection['identity']['contract']['pretraining']['identity']['contract']['architecture']
            args=author_args(args_cfg,architecture)
            args.world_model_backend='tssm' if phase=='released_tssm' else 'drama_mamba2'
            env=IR2RNavBatch(ImageFeaturesDB(args.img_ft_file,512),[by_id[i] for i in record['members']],
                args.connectivity_dir,{record['scan']:[record['members']]},num_robots=1,batch_size=1,
                angle_feat_size=4,seed=0,name='train_dev')
            if agent is None:
                seed_all(0);agent=GMapNavAgent(args,env,rank=0)
                if phase=='released_tssm':
                    payload=torch.load(cfg['released_checkpoint'],map_location='cpu',weights_only=True)
                    weights=payload['vln_bert']['state_dict']
                    if all(k.startswith('module.') for k in weights):weights={k[7:]:v for k,v in weights.items()}
                    agent.vln_bert.load_state_dict(weights,strict=True);del weights,payload
                else:
                    weights,selected_again=load_ir2r_selection(cfg)
                    require(selected_again==selection,'Selected best checkpoint changed during evaluation')
                    agent.vln_bert.vln_bert.load_state_dict(weights,strict=True);del weights
                agent.vln_bert.requires_grad_(False)
            rows=evaluate_tour(agent,env,record)
            data={'complete':True,'identity_sha256':identity_sha,'phase':phase,'tour':record,'rows':rows}
            guard()
            publish_tour(output,backup,phase,record,identity_sha,data)
            print('EVALUATION_TOUR',json.dumps({'phase':phase,'tour':record['tour_id'],**aggregate(rows)}),flush=True)
        del agent
        gc.collect();torch.cuda.empty_cache()
    guard()
    all_rows={phase:[row for tour in tours for row in read_tour(backup,phase,tour,identity_sha)['rows']] for phase in PHASES}
    result=comparison(all_rows[PHASES[0]],all_rows[PHASES[1]])
    write_json(output/'comparison.json',result);result_sha=verified_copy(output/'comparison.json',backup/'comparison.json')
    complete={'complete':True,'schema':SCHEMA,'instructions_per_model':2889,'tours_per_model':36,
        'comparison_sha256':result_sha,'delta':result['delta_drama_minus_released'],
        'limitation':LIMITATION,'selected_step':selection['selected_step']}
    write_json(output/'completion.json',complete);verified_copy(output/'completion.json',backup/'completion.json')
    guard();validate_results(cfg)
    print('EVALUATION_COMPLETE',json.dumps(complete),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--require-resume',action='store_true');args=parser.parse_args()
    run(json.loads(Path(args.config).read_text()),args.require_resume)


if __name__=='__main__':main()
