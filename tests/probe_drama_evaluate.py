#!/usr/bin/env python3
"""Engineering-only native GPU smoke: released TSSM and fresh Drama, two fit episodes.

Does not produce production completion/checkpoints or access train_dev/val_unseen.
Uses production evaluate_tour and compares every trajectory score to the author's
eval_metrics result, including aggregates. Run only after GPU handoff.
"""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import drama_evaluate as evaluation
import drama_pretrain as pre
from drama_ir2r import author_args,select_split,require


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pretrain-config',default='configs/drama-pretrain.json')
    parser.add_argument('--ir2r-config',default='configs/drama-ir2r.json')
    parser.add_argument('--eval-config',default='configs/drama-evaluate.json')
    parser.add_argument('--output',default='/home/xxl/memoir-disk-20261116/qa/evaluation/probe-v1')
    cli=parser.parse_args();os.chdir(ROOT)
    pre_cfg=json.loads(Path(cli.pretrain_config).read_text())
    train_cfg=json.loads(Path(cli.ir2r_config).read_text())
    eval_cfg=json.loads(Path(cli.eval_config).read_text())
    pre.configure_runtime(pre_cfg)
    output=Path(cli.output)
    storage={**pre_cfg,'run_dir':str(output),'backup_dir':str(output.parent/(output.name+'-unused-backup'))}
    mount=pre.verify_storage_mount(storage)
    def guard():
        pre.verify_storage_mount(storage,mount)
        pre.space_guard([storage['storage_mount'],storage['host_volume']],storage['min_free_gib'])
    guard();require(not output.exists(),'Fresh engineering probe refuses existing output')
    output.mkdir(parents=True)
    started=time.monotonic();source=pre.code_manifest()
    try:
        from nav_src.envs.r2r.data_utils import construct_instrs
        from nav_src.envs.r2r.iter_env import IR2RNavBatch
        from nav_src.iter_nav.agent_ir2r import GMapNavAgent
        from utils.data import ImageFeaturesDB
        require(pre.sha(eval_cfg['released_checkpoint'])==eval_cfg['released_checkpoint_sha256'],'Released SHA differs')
        split=json.loads(Path(train_cfg['split_manifest']).read_text())
        instructions=construct_instrs(str(Path(train_cfg['datasets'])/'R2R'/'annotations'),'r2r',['train'],'bert',512)
        all_tours=json.loads(Path(train_cfg['tours']).read_text())['train']
        items,tours=select_split(instructions,all_tours,split['train_fit']);by_id={x['instr_id']:x for x in items}
        scan=sorted(tours)[0];ids=tours[scan][0][:2];require(len(ids)==2,'Engineering tour needs two instructions')
        record={'tour_id':scan+'__engineering_prefix2','scan':scan,'members':ids,'seed':917}
        architecture={'d_model':pre_cfg['d_model'],'layers':pre_cfg['layers'],'d_state':pre_cfg['d_state'],
                      'headdim':pre_cfg['headdim'],'dropout':pre_cfg['drama_dropout']}
        result={'engineering_only':True,'production_completion':False,'scope':'train_fit author tour prefix of two episodes',
            'tour':record,'source_files':source,'runtime':pre.runtime_identity(),'phases':{}}
        for phase in ('released_tssm','random_drama'):
            guard();torch.cuda.reset_peak_memory_stats();before=time.monotonic()
            cfg={**train_cfg,'run_dir':str(output/'author-layout'/phase),'batch_size':1}
            args=author_args(cfg,architecture)
            args.world_model_backend='tssm' if phase=='released_tssm' else 'drama_mamba2'
            args.bert_ckpt_file=None if phase=='released_tssm' else pre_cfg['init_checkpoint']
            env=IR2RNavBatch(ImageFeaturesDB(args.img_ft_file,512),[by_id[i] for i in ids],args.connectivity_dir,
                {scan:[ids]},num_robots=1,batch_size=1,angle_feat_size=4,seed=0,name='engineering_train_fit')
            pre.seed_all(0);agent=GMapNavAgent(args,env,rank=0)
            if phase=='released_tssm':
                payload=torch.load(eval_cfg['released_checkpoint'],map_location='cpu',weights_only=True)
                weights=payload['vln_bert']['state_dict']
                if all(k.startswith('module.') for k in weights):weights={k[7:]:v for k,v in weights.items()}
                agent.vln_bert.load_state_dict(weights,strict=True);del payload,weights
            agent.vln_bert.requires_grad_(False)
            captured=[];native_rollout=agent.rollout
            def capture(*a,**kw):
                returned=native_rollout(*a,**kw);captured.extend(returned);return returned
            agent.rollout=capture
            rows=evaluation.evaluate_tour(agent,env,record)
            agent.results={item['instr_id']:item for item in captured}
            native_predictions=agent.get_results()
            native_metrics,native_rows=env.eval_metrics(native_predictions)
            own=evaluation.aggregate(rows)
            for new,old in [('SR','sr'),('SPL','spl'),('NE','nav_error'),('nDTW','nDTW')]:
                require(np.isclose(own[new],native_metrics[old],atol=1e-10,rtol=0),'Aggregate mismatch: '+new)
            for index,row in enumerate(rows):
                require(native_rows['instr_id'][index]==row['instr_id'],'Author result ordering differs')
                for key,value in row['scores'].items():
                    require(np.isclose(value,native_rows[key][index],atol=1e-10,rtol=0),'Per-trajectory mismatch: '+key)
            result['phases'][phase]={'strict_release_load':phase=='released_tssm','metrics':own,'rows':rows,
                'matches_author_eval_metrics':True,'seconds':time.monotonic()-before,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}
            print('ENGINEERING_EVAL_PHASE',json.dumps({k:v for k,v in result['phases'][phase].items() if k!='rows'}),flush=True)
            agent.rollout=native_rollout
            del agent,native_rollout,capture,env,captured,native_predictions
            gc.collect();torch.cuda.empty_cache()
        require(pre.code_manifest()==source,'Production source changed during engineering probe')
        result.update(passed=True,seconds=time.monotonic()-started)
        guard();pre.write_json(output/'result.json',result)
        print('ENGINEERING_EVALUATION_PASSED',str(output/'result.json'),flush=True)
    except BaseException:
        guard();pre.write_json(output/'failure.json',{'engineering_only':True,'passed':False,'traceback':traceback.format_exc()})
        raise


if __name__=='__main__':main()
