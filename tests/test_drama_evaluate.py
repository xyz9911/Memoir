import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import drama_evaluate as ev
import drama_pretrain as pre
from drama_checkpoint_store import CheckpointStore


def row(key,success,spl,ne,ndtw):
    return {'instr_id':key,'scan':'scan','scores':{
        'success':success,'spl':spl,'nav_error':ne,'nDTW':ndtw}}


class EvaluationTests(unittest.TestCase):
    def test_paired_deltas_have_explicit_percent_point_and_meter_units(self):
        baseline=[row('a',0,0,5,.2),row('b',1,.6,1,.7)]
        candidate=[row('b',1,.8,.5,.9),row('a',1,.5,2,.5)]
        result=ev.comparison(baseline,candidate)
        self.assertEqual(result['delta_drama_minus_released']['SR'],50.)
        self.assertAlmostEqual(result['delta_drama_minus_released']['SPL'],35.)
        self.assertEqual(result['delta_drama_minus_released']['NE'],-1.75)
        self.assertTrue(result['goal_sr_plus5_spl_nondecreasing'])
        self.assertEqual([r['instr_id'] for r in result['paired']],['a','b'])
        with self.assertRaisesRegex(RuntimeError,'different instruction sets'):
            ev.comparison(baseline,candidate[:1])

    def test_complete_tour_preserves_internal_memory_and_does_not_repeat_sentinel(self):
        class Environment:
            gt_trajs={key:('scan',['start','goal']) for key in ('a','b','c')}
            def reset_epoch(self,shuffle):self.index=0
            def check_last(self):return self.index==3
            def _eval_item(self,scan,path,gt):return {'success':1.,'spl':.5,'nav_error':1.,'nDTW':.6}
        class Agent:
            vln_bert=SimpleNamespace(eval=lambda:None)
            def rollout(self,extended_memory):
                self.gmaps.append(self.env.index)
                self.env.index+=1
                return [{'instr_id':'abc'[self.env.index-1],'path':[['start'],['goal']]}]
        env=Environment();agent=Agent();record={'members':['a','b','c'],'seed':917}
        rows=ev.evaluate_tour(agent,env,record)
        self.assertEqual(len(rows),3);self.assertEqual(agent.gmaps,[0,1,2]);self.assertEqual(env.index,3)

    def test_tour_resume_requires_identity_coverage_and_readback_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            record={'tour_id':'scan__0000','members':['a'],'seed':17}
            path,marker=ev.artifact_paths(directory,'released_tssm',record['tour_id'])
            self.assertIsNone(ev.read_tour(directory,'released_tssm',record,'identity'))
            path.parent.mkdir(parents=True)
            pre.write_json(path,{'identity_sha256':'identity','phase':'released_tssm',
                'tour':record,'complete':True,'rows':[row('a',1,.5,1,.6)]})
            with self.assertRaisesRegex(RuntimeError,'Incomplete tour artifact'):
                ev.read_tour(directory,'released_tssm',record,'identity')
            marker.write_text(pre.sha(path)+'\n')
            self.assertTrue(ev.read_tour(directory,'released_tssm',record,'identity')['complete'])
            with self.assertRaisesRegex(RuntimeError,'identity differs'):
                ev.read_tour(directory,'released_tssm',record,'wrong')
            path.write_text('{}')
            with self.assertRaisesRegex(RuntimeError,'checksum differs'):
                ev.read_tour(directory,'released_tssm',record,'identity')

    def test_ir2r_reader_uses_actual_compact_schema_and_completion_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            run,backup=Path(directory)/'local',Path(directory)/'backup';run.mkdir();backup.mkdir()
            pre.atomic_torch(backup/'frozen-base.pt',{'encoder':torch.ones(2)})
            identity={'frozen_base_sha256':pre.sha(backup/'frozen-base.pt'),
                'contract':{'schema':ev.IR2R_SCHEMA,'config':{'minimum_updates':20000},'delta_names':['world']}}
            for root in (run,backup):pre.write_json(root/'identity.json',identity)
            completion={'complete':True,'actual_updates':20003,'best_step':20003}
            pre.write_json(run/'completion.json',completion)
            store=CheckpointStore(run/'checkpoints',backup/'checkpoints',keep_local=1,keep_backup=1)
            state={'schema':ev.IR2R_SCHEMA,'identity_sha256':pre.json_sha(identity),'step':20003,
                'progress':{'completed_updates':20003,'best_step':20003},'trainable_state':{'world':torch.ones(1)}}
            store.save(state,{},step=20003,is_best=True)
            cfg={'ir2r_run':str(run),'ir2r_backup':str(backup)}
            weights,metadata=ev.load_ir2r_selection(cfg)
            self.assertEqual(set(weights),{'encoder','world'});self.assertEqual(metadata['selected_step'],20003)
            bad=copy.deepcopy(completion);bad['actual_updates']=19999;pre.write_json(run/'completion.json',bad)
            with self.assertRaisesRegex(RuntimeError,'complete production'):
                ev.load_ir2r_selection(cfg)

    def test_interrupted_uncommitted_tour_replays_but_verified_backup_recovers_local(self):
        with tempfile.TemporaryDirectory() as directory:
            run,backup=Path(directory)/'local',Path(directory)/'backup'
            record={'tour_id':'scan__0000','members':['a'],'seed':17};phase='selected_drama'
            path,marker=ev.artifact_paths(backup,phase,record['tour_id']);path.parent.mkdir(parents=True)
            path.write_text('interrupted-uncommitted-output')
            self.assertIsNone(ev.read_tour(backup,phase,record,'identity',allow_uncommitted=True))
            data={'identity_sha256':'identity','phase':phase,'tour':record,'complete':True,'rows':[row('a',1,.5,1,.6)]}
            ev.publish_tour(run,backup,phase,record,'identity',data)
            local,local_marker=ev.artifact_paths(run,phase,record['tour_id']);local.write_text('broken-local')
            ev.restore_local_tour(run,backup,phase,record,'identity')
            self.assertEqual(ev.read_tour(run,phase,record,'identity'),data)
            with self.assertRaisesRegex(RuntimeError,'overwrite a committed'):
                ev.publish_tour(run,backup,phase,record,'identity',data)
            path.unlink();marker.unlink()
            with self.assertRaisesRegex(RuntimeError,'Committed local tour exists without backup'):
                ev.publish_tour(run,backup,phase,record,'identity',data)


if __name__=='__main__':unittest.main()
