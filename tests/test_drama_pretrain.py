import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import drama_pretrain as pre
import drama_pipeline as pipeline
import drama_mount_guard as mount_guard
from drama_checkpoint_store import CheckpointStore
from vlnbert.drama_mamba2 import DramaMamba2Prior, freeze_for_world_model_pretraining

class TinyWorld(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace()
        self.embeddings=nn.Linear(8,8);self.lang_encoder=nn.Linear(8,8)
        self.img_embeddings=nn.ModuleDict({'shared':self.embeddings})
        self.lang_init=nn.Linear(8,6)
        self.prior=DramaMamba2Prior(2,False,8,6,d_model=16,n_layers=2,d_state=4,headdim=8,dropout=.1)
        self.posterior=nn.Linear(14,4)
        self.to_vis_embed=nn.Linear(8,4);self.to_reward=nn.Linear(8,1);self.obs_transform=nn.Linear(8,4)
    def distribution(self,s):
        m,v=s.chunk(2,-1);return torch.distributions.Normal(m,F.softplus(v)+.1)
    def forward_posterior(self,d,o):
        p=self.distribution(self.posterior(torch.cat((d,o),-1)))
        return {'dist':p,'stoch':p.rsample()}
    def forward_transit(self,prev,obs,text,mask,past_key_values=None):
        stats,d,c=self.prior(prev,text,mask,past_key_values)
        prior=self.distribution(stats);post=self.forward_posterior(d,obs)
        return dict(prior_dist=prior,posterior_dist=post['dist'],posterior_stoch=post['stoch'],
                    prior_stoch=prior.rsample(),deter=d,past_key_values=c)
    def forward_imagine(self,prev,text,mask,past_key_values=None):
        stats,d,c=self.prior(prev,text,mask,past_key_values);p=self.distribution(stats)
        return dict(dist=p,stoch=p.rsample(),deter=d,past_key_values=c)
    def forward_reconstruct(self,s):return dict(img_embed=self.to_vis_embed(s),reward=self.to_reward(s))

def batch():
    return {'language':torch.randn(2,3,8),'text_mask':torch.tensor([[1,1,1],[1,1,0]],dtype=torch.bool),
            'visual':torch.randn(2,5,8),'mask':torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool),
            'distance':torch.rand(2,5)}

class TrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);pre.seed_all(3)
    def optimizer_config(self):
        return {'lr':5e-5,'betas':[.9,.98],'weight_decay':.01,'adam_eps':1e-6,
                'grad_clip':5.,'warmup_steps':100,'steps':6000}
    def test_real_mamba_loss_finite_backward_and_frozen_shared_encoders(self):
        m=TinyWorld();freeze_for_world_model_pretraining(m)
        loss,pieces=pre.wm_loss(m,batch(),3);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(set(pieces),{'kl','overshoot_kl','visual_nce','reward_mse'})
        for name,p in m.named_parameters():
            if name.startswith(pre.WM_PREFIXES):
                self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            else:self.assertIsNone(p.grad,name);self.assertFalse(p.requires_grad,name)
        self.assertFalse(m.embeddings.training)
    def test_padding_never_enters_loss_or_recurrence_of_valid_rows(self):
        m=TinyWorld().eval();b=batch();alter={k:v.clone() for k,v in b.items()}
        alter['visual'][1,3:]=10000;alter['distance'][1,3:]=-10000;alter['language'][1,2]=10000
        pre.seed_all(17);a,_=pre.wm_loss(m,b,3)
        pre.seed_all(17);z,_=pre.wm_loss(m,alter,3)
        torch.testing.assert_close(a,z,rtol=0,atol=0)
    def test_compact_state_restores_next_optimizer_update_and_sampler(self):
        cfg=self.optimizer_config()
        m=TinyWorld();freeze_for_world_model_pretraining(m);opt=pre.build_optimizer(m,cfg)
        b=batch();sampler=pre.Sampler(9)
        def update(model,optimizer,step):
            optimizer.zero_grad();loss,_=pre.wm_loss(model,b,2);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
            for group in optimizer.param_groups:group['lr']=pre.scheduled_lr(step,cfg)
            optimizer.step();return float(loss)
        update(m,opt,1);sampler.take(4)
        frozen={k:v.clone() for k,v in m.state_dict().items() if not k.startswith(pre.WM_PREFIXES)}
        delta={k:v.clone() for k,v in m.state_dict().items() if k.startswith(pre.WM_PREFIXES)}
        state=dict(trainable_state=delta,optimizer=copy.deepcopy(opt.state_dict()),rng=pre.rng_state(),sampler=sampler.state(),scheduler=pre.scheduler_state(1,cfg))
        expected_indices=sampler.take(12);expected_loss=update(m,opt,2)
        restored=TinyWorld();freeze_for_world_model_pretraining(restored);restored.load_state_dict({**frozen,**delta})
        restored_opt=pre.build_optimizer(restored,cfg)
        restored_opt.load_state_dict(state['optimizer']);sampler2=pre.Sampler(9,state['sampler']);pre.set_rng(state['rng'])
        pre.validate_scheduler(state['scheduler'],1,cfg,restored_opt)
        self.assertEqual(sampler2.take(12),expected_indices);self.assertEqual(update(restored,restored_opt,2),expected_loss)
        for key,v in m.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[key],rtol=0,atol=0)
    def test_author_schedule_boundaries_and_resume_mismatch(self):
        cfg=self.optimizer_config();m=TinyWorld();opt=pre.build_optimizer(m,cfg)
        self.assertAlmostEqual(pre.scheduled_lr(1,cfg),5e-7)
        self.assertEqual(pre.scheduled_lr(100,cfg),5e-5)
        self.assertEqual(pre.scheduled_lr(3050,cfg),2.5e-5)
        self.assertEqual(pre.scheduled_lr(6000,cfg),1e-8)
        pre.validate_scheduler(pre.scheduler_state(0,cfg),0,cfg,opt)
        with self.assertRaisesRegex(RuntimeError,'scheduler state differs'):
            pre.validate_scheduler(pre.scheduler_state(100,cfg),99,cfg,opt)
        no_decay={id(p) for g in opt.param_groups if not g['weight_decay'] for p in g['params']}
        for name,p in m.named_parameters():
            if getattr(p,'_no_weight_decay',False) or 'bias' in name:self.assertIn(id(p),no_decay)
    def test_horizon_one_has_no_overshoot(self):
        m=TinyWorld();_,pieces=pre.wm_loss(m,batch(),1)
        self.assertEqual(float(pieces['overshoot_kl']),0.)
    def test_validation_rng_isolated_even_on_exception(self):
        s=pre.rng_state();expected=torch.rand(3);pre.set_rng(s)
        with self.assertRaisesRegex(RuntimeError,'test'):
            with pre.isolated_rng(999):
                torch.rand(100);np.random.rand(10);raise RuntimeError('test')
        torch.testing.assert_close(torch.rand(3),expected,rtol=0,atol=0)
    def test_disk_guard_checks_physical_host_and_extra_snapshot_budget(self):
        with tempfile.TemporaryDirectory() as d,patch.object(pre.shutil,'disk_usage',return_value=SimpleNamespace(free=13*1024**3)):
            pre.space_guard([d],12)
            with self.assertRaisesRegex(RuntimeError,'Insufficient'):pre.space_guard([d],12,2*1024**3)
    def test_host_mount_parent_fallback_rejected(self):
        with patch.object(pre,'backup_mount_identity',return_value={'mount_point':'/','filesystem':'ext4'}):
            with self.assertRaisesRegex(RuntimeError,'explicit mounted'):pre.verify_host_volume('/mnt/e')
    def test_windows_mountinfo_source_octal_escape_is_canonical(self):
        fixture='1 0 0:1 / /mnt/g rw - 9p G:\\134 rw\n'
        with patch.object(Path,'is_dir',return_value=True),patch.object(Path,'is_file',return_value=True),patch.object(Path,'read_text',return_value=fixture):
            identity=mount_guard.backup_mount_identity(Path('/mnt/g'),backend='filesystem')
        self.assertEqual(identity['source'],'G:\\')
        self.assertEqual(identity['mount_point'],'/mnt/g')
    def test_runtime_binds_resolved_hf_configuration_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'config.json';path.write_text('{"hidden_size":768}')
            cached=Mock(return_value=str(path))
            hub=SimpleNamespace(cached_file=cached)
            with patch.dict(sys.modules,{'transformers.utils.hub':hub}),patch.object(pre.metadata,'version',return_value='fixed'),patch.object(pre,'find_spec',return_value=None):
                before=pre.runtime_identity()
                cached.assert_called_with('bert-base-uncased','config.json',local_files_only=True)
                self.assertEqual(before['bert_config']['sha256'],pre.sha(path))
                path.write_text('{"hidden_size":512}')
                after=pre.runtime_identity()
            self.assertNotEqual(before['bert_config']['sha256'],after['bert_config']['sha256'])
            self.assertTrue(before['deterministic_algorithms']);self.assertFalse(before['matmul_tf32'])
    def test_missing_or_changed_numerics_rejected_without_cfg_mutation(self):
        cfg={}
        with self.assertRaisesRegex(RuntimeError,'Numerical runtime'):pre.configure_runtime(cfg)
        self.assertEqual(cfg,{})
    def test_cache_rejects_base_source_split_or_membership_identity_change(self):
        with tempfile.TemporaryDirectory() as d:
            cfg={'cache':str(Path(d)/'cache.hdf5')};identity={'inputs':{'base':'original'},'engineering_limit':None}
            marker=Path(cfg['cache']).with_suffix('.manifest.json')
            pre.write_json(marker,{'complete':True,'identity':identity,'identity_sha256':pre.json_sha(identity)})
            with patch.object(pre,'cache_inputs',return_value={'inputs':{'base':'original'}}):
                self.assertTrue(pre.validate_cache_contract(cfg)['complete'])
            with patch.object(pre,'cache_inputs',return_value={'inputs':{'base':'changed'}}):
                with self.assertRaisesRegex(RuntimeError,'identity differs'):pre.validate_cache_contract(cfg)
    def test_pipeline_rechecks_completed_marker_and_cache_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            cache=Path(d)/'cache.hdf5';cache.write_bytes(b'sealed-bytes')
            cfg={'cache':str(cache)};marker=cache.with_suffix('.manifest.json')
            content={'complete':True,'sha256':pre.sha(cache)};pre.write_json(marker,content)
            digest=pre.sha(marker)
            with patch.object(pipeline,'validate_cache_contract',return_value=content):
                self.assertEqual(pipeline.validate_completed_stage('cache',cfg,digest),digest)
                cache.write_bytes(b'corrupt')
                with self.assertRaisesRegex(RuntimeError,'payload checksum'):pipeline.validate_completed_stage('cache',cfg,digest)
            pre.write_json(marker,{'complete':False})
            with self.assertRaisesRegex(RuntimeError,'marker changed'):pipeline.validate_completed_stage('cache',cfg,digest)
    def test_loop_storage_refuses_unmounted_parent_before_write(self):
        cfg={'storage_mount':'/not/mounted','storage_image':'/mnt/g/image.ext4','host_volume':'/mnt/g','host_volume_source':'G:\\','min_free_gib':12}
        with patch.object(pre,'verify_host_volume',return_value={'source':'G:\\'}),patch.object(pre,'backup_mount_identity',return_value={'source':'/dev/sdd','mount_point':'/','filesystem':'ext4'}):
            with self.assertRaisesRegex(RuntimeError,'parent filesystem fallback'):pre.verify_storage_mount(cfg)
    def test_selected_export_is_full_base_plus_verified_best_delta(self):
        with tempfile.TemporaryDirectory() as d:
            local=Path(d)/'local';backup=Path(d)/'backup';local.mkdir();backup.mkdir()
            pre.atomic_torch(local/'frozen-base.pt',{'encoder':torch.ones(2)})
            pre.verified_copy(local/'frozen-base.pt',backup/'frozen-base.pt')
            identity={'frozen_base_sha256':pre.sha(local/'frozen-base.pt')}
            pre.write_json(local/'identity.json',identity)
            store=CheckpointStore(local/'checkpoints',backup/'checkpoints',keep_local=1,keep_backup=1)
            state={'identity_sha256':pre.json_sha(identity),'step':5,'best_dev':.4,'trainable_state':{'wm':torch.tensor([2.])}}
            store.save(state,{},step=5,is_best=True)
            weights,info=pre.load_selected_weights(local,backup)
            self.assertEqual(set(weights),{'encoder','wm'});self.assertEqual(info['selected_step'],5)
            (backup/'frozen-base.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError,'Frozen base'):pre.load_selected_weights(local,backup)

if __name__=='__main__':unittest.main()
