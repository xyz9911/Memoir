#!/usr/bin/env python3
"""Frozen-encoder, GT-trajectory world-model pretraining; no navigation/SAP loss.

The frozen base is saved once. Every compact snapshot plus this SHA-bound base
fully reconstructs the model, optimizer, sampler and RNG. Local and backup
snapshots are committed/read-back verified before retention. No implicit resume.
"""
import argparse
from contextlib import contextmanager
import hashlib
from importlib import metadata
from importlib.util import find_spec
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drama_checkpoint_store import _atomic_bytes, _json_bytes
from drama_mount_guard import MountedCheckpointStore, backup_mount_identity, validate_backup_root

SCHEMA = 'memoir-drama-frozen-wm-v1'
WM_PREFIXES = ('lang_init.', 'prior.', 'posterior.', 'to_vis_embed.', 'to_reward.', 'obs_transform.')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def write_json(path, value):
    _atomic_bytes(Path(path), _json_bytes(value))

def space_guard(paths, min_free_gib, additional=0):
    required = int(min_free_gib * 1024**3) + additional
    for path in paths:
        if not Path(path).exists():
            raise RuntimeError(f'Space guard mount missing: {path}')
        free = shutil.disk_usage(path).free
        if free < required:
            raise RuntimeError(f'Insufficient space at {path}: {free} < {required} bytes; training stops without pruning')

def verify_host_volume(path,expected=None):
    identity=backup_mount_identity(Path(path),backend='filesystem')
    if Path(identity['mount_point']).resolve()!=Path(path).resolve() or identity['filesystem'] not in ('9p','drvfs'):
        raise RuntimeError('Host volume must be an explicit mounted Windows drive, not a Linux parent fallback')
    if expected is not None and identity!=expected:
        raise RuntimeError('Windows host volume mount identity changed')
    return identity

def verify_storage_mount(cfg,expected=None):
    if cfg.get('min_free_gib',0)<12:raise RuntimeError('Production storage reserve must be at least 12 GiB')
    mount=Path(cfg['storage_mount']).resolve();image=Path(cfg['storage_image']).resolve()
    host=Path(cfg['host_volume']).resolve()
    host_identity=verify_host_volume(host)
    if host_identity['source'].casefold()!=cfg['host_volume_source'].casefold():
        raise RuntimeError('Selected Windows host volume source differs')
    identity=backup_mount_identity(mount,backend='filesystem')
    source=Path(identity['source'])
    if (Path(identity['mount_point']).resolve()!=mount or identity['filesystem']!='ext4'
            or not re.fullmatch(r'loop[0-9]+',source.name) or str(source.parent)!='/dev'):
        raise RuntimeError('Dedicated ext4 loop storage is absent; refusing parent filesystem fallback')
    backing=Path('/sys/block')/source.name/'loop'/'backing_file'
    if not backing.is_file() or Path(backing.read_text().strip()).resolve()!=image:
        raise RuntimeError('Loop device backing file is not the selected storage image')
    if not image.is_file() or not image.is_relative_to(host):
        raise RuntimeError('Storage image is not a file on the selected host volume')
    for key in ('run_dir','backup_dir','cache'):
        if key in cfg and not Path(cfg[key]).resolve().is_relative_to(mount):
            raise RuntimeError(f'{key} lies outside the selected ext4 mount')
    identity['storage_image']=str(image)
    if expected is not None and expected!=identity:raise RuntimeError('Dedicated storage mount identity changed')
    return identity

def model_args(cfg):
    return SimpleNamespace(bert_ckpt_file=cfg['init_checkpoint'], tokenizer='bert',
        image_feat_size=512, angle_feat_size=4, obj_feat_size=0,
        num_l_layers=9, num_pano_layers=2, num_x_layers=4, graph_sprels=True,
        fusion='dynamic', fix_lang_embedding=True, fix_pano_embedding=True,
        fix_local_branch=False, freeze_encoders=True, initialize_drama_world=True,
        world_model_backend='drama_mamba2', drama_d_model=cfg.get('d_model',512),
        drama_layers=cfg.get('layers',2), drama_d_state=cfg.get('d_state',16),
        drama_headdim=cfg.get('headdim',64), drama_dropout=cfg.get('drama_dropout',.1))

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def configure_runtime(cfg=None):
    numerics={'deterministic_algorithms':True,'cudnn_deterministic':True,'cudnn_benchmark':False,
              'matmul_tf32':False,'cudnn_tf32':False,'cublas_workspace_config':':4096:8'}
    if cfg is not None and cfg.get('numerics')!=numerics:
        raise RuntimeError('Numerical runtime configuration differs from the verified deterministic mode')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if os.environ['CUBLAS_WORKSPACE_CONFIG']!=':4096:8':
        raise RuntimeError('CUBLAS_WORKSPACE_CONFIG must be :4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def make_model(cfg, device='cuda'):
    configure_runtime(cfg)
    from vlnbert.vlnbert_init import get_vlnbert_models
    from vlnbert.drama_mamba2 import freeze_for_world_model_pretraining
    seed_all(cfg['seed'])
    model = get_vlnbert_models(model_args(cfg)).to(device)
    names = freeze_for_world_model_pretraining(model)
    expected = {n for n, p in model.named_parameters() if n.startswith(WM_PREFIXES)}
    assert set(names) == expected, 'World model trainable whitelist differs'
    assert all(p.requires_grad == (n in expected) for n,p in model.named_parameters())
    return model

def rng_state():
    s=np.random.get_state()
    return {'python':random.getstate(), 'numpy':(s[0],torch.tensor(s[1].astype(np.int64)),s[2],s[3],s[4]),
            'cpu':torch.get_rng_state(), 'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def set_rng(s):
    random.setstate(s['python']); n=s['numpy']
    np.random.set_state((n[0],n[1].numpy().astype(np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s['cpu'])
    if s['cuda']: torch.cuda.set_rng_state_all(s['cuda'])

@contextmanager
def isolated_rng(seed):
    state=rng_state()
    try:
        seed_all(seed); yield
    finally:
        set_rng(state)

def cpu_tree(v):
    if torch.is_tensor(v): return v.detach().cpu().clone()
    if isinstance(v,dict): return {k:cpu_tree(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return type(v)(cpu_tree(x) for x in v)
    return v

def tensor_bytes(value):
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if isinstance(value,dict): return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)): return sum(tensor_bytes(v) for v in value)
    return 0

def code_manifest():
    repo=Path(__file__).resolve().parents[1]
    return {str(p.relative_to(repo)):sha(p) for folder in ('vlnbert','utils','nav_src','pretrain','scripts')
            for p in sorted((repo/folder).rglob('*.py'))}

def runtime_identity():
    configure_runtime()
    from transformers.utils.hub import cached_file
    # Same config key as vlnbert_init; force local resolution even outside launcher.
    bert_config=Path(cached_file('bert-base-uncased','config.json',local_files_only=True)).resolve()
    simulator=find_spec('MatterSim')
    return {'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,
        'transformers':metadata.version('transformers'),'h5py':h5py.__version__,
        'bert_config':{'resolved_path':str(bert_config),'sha256':sha(bert_config),
                       'json_sha256':json_sha(json.loads(bert_config.read_text()))},
        'mattersim':{'path':simulator.origin,'sha256':sha(simulator.origin)} if simulator else None,
        'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'pythonhashseed':os.environ.get('PYTHONHASHSEED'),'cublas_workspace':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}

def optimizer_contract(cfg):
    return {'implementation':'pretrain.optim.adamw.AdamW','lr':cfg['lr'],
        'betas':list(cfg['betas']),'weight_decay':cfg['weight_decay'],
        'eps':cfg['adam_eps'],'correct_bias':True,'grad_clip':cfg['grad_clip'],
        'no_decay':'author bias/LayerNorm names plus Mamba _no_weight_decay parameters',
        'schedule':'pretrain.optim.sched.get_lr_sched: warmup_linear, floor 1e-8',
        'warmup_steps':cfg['warmup_steps'],'total_steps':cfg['steps']}

def build_optimizer(model,cfg):
    from pretrain.optim.adamw import AdamW
    decay=[];no_decay=[]
    for name,parameter in model.named_parameters():
        if not parameter.requires_grad:continue
        skip=getattr(parameter,'_no_weight_decay',False) or any(k in name for k in ('bias','LayerNorm.bias','LayerNorm.weight'))
        (no_decay if skip else decay).append(parameter)
    return AdamW([{'params':decay,'weight_decay':cfg['weight_decay']},
        {'params':no_decay,'weight_decay':0.}],lr=cfg['lr'],betas=tuple(cfg['betas']),eps=cfg['adam_eps'])

def scheduled_lr(step,cfg):
    from pretrain.optim.sched import get_lr_sched
    if not 0<=step<=cfg['steps'] or not 0<cfg['warmup_steps']<cfg['steps']:
        raise RuntimeError('Invalid optimizer update or warmup budget')
    return get_lr_sched(step,SimpleNamespace(learning_rate=cfg['lr'],
        warmup_steps=cfg['warmup_steps'],num_train_steps=cfg['steps']))

def scheduler_state(step,cfg):
    return {'completed_updates':step,'contract':optimizer_contract(cfg),
            'last_lr':scheduled_lr(step,cfg) if step else cfg['lr']}

def validate_scheduler(state,step,cfg,optimizer):
    expected=scheduler_state(step,cfg)
    if state!=expected or any(g['lr']!=expected['last_lr'] for g in optimizer.param_groups):
        raise RuntimeError('Strict resume scheduler state differs')

def cache_inputs(cfg):
    configure_runtime(cfg)
    paths=[cfg['init_checkpoint'],cfg['feature_file'],cfg['annotations'],cfg['tours'],cfg['split_manifest']]
    paths+=list(map(str,sorted(Path(cfg['connectivity']).glob('*_connectivity.json'))))
    annotations=json.loads(Path(cfg['annotations']).read_text())
    tours=json.loads(Path(cfg['tours']).read_text())['train']
    members=sorted({i for ts in tours.values() for t in ts for i in t})
    annotation_ids=sorted(f"{r['path_id']}_{i}" for r in annotations for i in range(len(r['instructions'])))
    return {'inputs':{str(Path(p).resolve()):sha(p) for p in paths},
            'tour_membership':{'count':len(members),'sha256':json_sha(members),
                'excluded_annotation_ids':sorted(set(annotation_ids)-set(members)),
                'rule':'official train tour members only; retain all author tours without added singleton tours'},
            'encoder_model':{k:cfg.get(k) for k in ('seed','d_model','layers','d_state','headdim')},
            'source_files':code_manifest(), 'runtime':runtime_identity(),'storage_dtype':'float32',
            'trajectory_orientation':'initial heading then previous outgoing candidate pointId; native discrete angles',
            'distance_target':'distance from GT next viewpoint to final goal, clipped to last viewpoint'}

def validate_cache_contract(cfg):
    manifest=json.loads(Path(cfg['cache']).with_suffix('.manifest.json').read_text())
    expected=cache_inputs(cfg);expected['engineering_limit']=None
    if (not manifest.get('complete') or manifest['identity']!=expected or
            manifest['identity_sha256']!=json_sha(expected)):
        raise RuntimeError('Frozen cache identity differs from current source/base/split/tour membership')
    return manifest


@torch.no_grad()
def encode_trajectory(model,env,helper,ex):
    """Encode one native GT trajectory in memory, using the sealed-cache semantics."""
    if model.embeddings.training or model.lang_encoder.training or model.img_embeddings.training:
        raise RuntimeError('Encoder must be in eval mode')
    ids=torch.tensor(ex['instr_encoding'],device=next(model.parameters()).device).unsqueeze(0)
    language=model.forward_text(ids,torch.ones_like(ids,dtype=torch.bool))[0].cpu().numpy()
    env.batch=[[ex]]; env.tour_ended[:]=False
    heading=float(ex['heading']); elevation=0.; visuals=[]; targets=[]
    for t,vp in enumerate(ex['path']):
        env.env.sims[0][0].newEpisode([ex['scan']],[vp],[heading],[elevation])
        ob=env._get_obs(t=t)[0][0]
        inputs=helper._panorama_feature_variable([ob]); helper.store_obs.clear()
        pano,mask=model.forward_panorama_per_step(inputs['view_img_fts'],None,inputs['loc_fts'],
            inputs['nav_types'],inputs['view_lens'],None)
        mean=(pano*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
        visuals.append(mean[0].cpu().numpy())
        nxt=ex['path'][min(t+1,len(ex['path'])-1)]
        targets.append(env.shortest_distances[ex['scan']][nxt][ex['path'][-1]])
        if t+1<len(ex['path']):
            cand=next((c for c in ob['candidate'] if c['viewpointId']==nxt),None)
            if cand is None: raise RuntimeError('GT edge absent in native candidate graph')
            heading=(cand['pointId']%12)*math.pi/6
            elevation=(cand['pointId']//12-1)*math.pi/6
    return {'language':language, 'visual':np.stack(visuals).astype(np.float32),
            'distance':np.asarray(targets,dtype=np.float32)}

def build_cache(cfg, limit=None):
    """Cache only fit/dev source-train instructions, with exact frozen encoders."""
    configure_runtime(cfg)
    from nav_src.envs.r2r.data_utils import construct_instrs
    from nav_src.envs.r2r.iter_env import IR2RNavBatch
    from nav_src.iter_nav.agent_ir2r import GMapNavAgent
    from utils.data import ImageFeaturesDB
    storage_mount=verify_storage_mount(cfg)
    out=Path(cfg['cache']); out.parent.mkdir(parents=True,exist_ok=True)
    manifest_path=out.with_suffix('.manifest.json')
    host_mount=verify_host_volume(cfg['host_volume'])
    identity=cache_inputs(cfg); identity['engineering_limit']=limit
    digest=json_sha(identity)
    if out.exists() and manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['identity_sha256']!=digest: raise RuntimeError('Cache identity differs; use new path')
        if old.get('complete'):
            if sha(out)!=old['sha256']: raise RuntimeError('Cache checksum failed')
            return old
    elif out.exists(): raise RuntimeError('Cache exists without identity')
    splits=json.loads(Path(cfg['split_manifest']).read_text())
    fit=set(splits['train_fit']); dev=set(splits['train_dev'])
    if fit & dev or not fit or not dev: raise RuntimeError('Invalid disjoint split manifest')
    data=construct_instrs(str(Path(cfg['annotations']).parent),'r2r',['train'],'bert',512,is_test=False)
    tours=json.loads(Path(cfg['tours']).read_text())['train']
    env=IR2RNavBatch(ImageFeaturesDB(cfg['feature_file'],512),data,cfg['connectivity'],tours,
                    num_robots=1,batch_size=1,angle_feat_size=4,seed=cfg['seed'],name='train')
    model=make_model(cfg); model.requires_grad_(False); model.eval()
    helper=object.__new__(GMapNavAgent); helper.args=model_args(cfg); helper.store_obs={}
    members={i for ts in tours.values() for t in ts for i in t}
    selected=[(ex,'train_fit' if ex['scan'] in fit else 'train_dev') for ex in data
              if ex['scan'] in fit|dev and ex['instr_id'] in members]
    if limit is not None: selected=selected[:limit]
    write_json(manifest_path,{'complete':False,'identity_sha256':digest,'identity':identity})
    started=time.time(); counts={'train_fit':0,'train_dev':0}
    with h5py.File(out,'a') as f, torch.no_grad():
        for k,(ex,split) in enumerate(selected):
            key=ex['instr_id']; counts[split]+=1
            if key in f and f[key].attrs.get('ready',False): continue
            if key in f: del f[key]
            if k%50==0:
                verify_storage_mount(cfg,storage_mount)
                verify_host_volume(cfg['host_volume'],host_mount)
                space_guard([out.parent,cfg['host_volume']],cfg['min_free_gib'])
                print('CACHE_PROGRESS',json.dumps({'done':k,'total':len(selected),'seconds':time.time()-started}),flush=True)
            encoded=encode_trajectory(model,env,helper,ex)
            g=f.create_group(key)
            for name,value in encoded.items():
                g.create_dataset(name,data=value.astype(np.float32),compression='lzf')
            g.attrs['split']=split;g.attrs['scan']=ex['scan'];g.attrs['ready']=True
            f.flush()
    result={'complete':True,'identity_sha256':digest,'identity':identity,'sha256':sha(out),
            'bytes':out.stat().st_size,'counts':counts,'seconds':time.time()-started}
    write_json(manifest_path,result)
    return result

class CachedTrajectories:
    def __init__(self,path,split,allow_engineering=False):
        self.path=Path(path);self.manifest=json.loads(self.path.with_suffix('.manifest.json').read_text())
        if not self.manifest['complete'] or sha(self.path)!=self.manifest['sha256']: raise RuntimeError('Unsealed/corrupt cache')
        if self.manifest['identity'].get('engineering_limit') is not None and not allow_engineering:
            raise RuntimeError('Engineering cache refused for full training')
        self.file=h5py.File(self.path,'r')
        self.ids=sorted(k for k in self.file if self.file[k].attrs['split']==split)
        if not self.ids: raise RuntimeError(f'Empty {split} cache')
    def batch(self,indices,device):
        rows=[self.file[self.ids[i]] for i in indices]
        b=len(rows);nt=max(r['language'].shape[0] for r in rows); nv=max(r['visual'].shape[0] for r in rows)
        lang=torch.zeros(b,nt,768,device=device);vis=torch.zeros(b,nv,768,device=device)
        txt_mask=torch.zeros(b,nt,dtype=torch.bool,device=device);mask=torch.zeros(b,nv,dtype=torch.bool,device=device)
        distance=torch.zeros(b,nv,device=device)
        for i,r in enumerate(rows):
            l,v=r['language'][:],r['visual'][:]
            lang[i,:len(l)]=torch.from_numpy(l).to(device);vis[i,:len(v)]=torch.from_numpy(v).to(device)
            txt_mask[i,:len(l)]=True;mask[i,:len(v)]=True
            distance[i,:len(v)]=torch.from_numpy(r['distance'][:]).to(device)
        return {'language':lang,'text_mask':txt_mask,'visual':vis,'mask':mask,'distance':distance}

def wm_loss(model,batch,horizon=5,reward_weight=.1,visual_weight=1.,kl_weight=2.,overshoot_kl_weight=2.,
            *, contrastive_pool=None, first_label=0, normalizers=None):
    """KL + cross-batch/time InfoNCE + GT next-goal-distance MSE, posterior/prior/overshoot."""
    text,tm,visual,mask,dist=(batch[k] for k in ('language','text_mask','visual','mask','distance'))
    B,T,_=visual.shape
    labels=torch.full((B,T),-1,device=visual.device,dtype=torch.long)
    labels[mask]=torch.arange(int(mask.sum()),device=visual.device) + first_label
    pool=contrastive_pool if contrastive_pool is not None else F.normalize(model.obs_transform(visual[mask]),dim=-1)
    deter=model.lang_init(text[:,0]);post=model.forward_posterior(deter,visual[:,0])
    prev=torch.cat((post['stoch'],deter),-1).unsqueeze(1);cache=None
    kl_sum=visual.new_zeros(());visual_sum=visual.new_zeros(());reward_sum=visual.new_zeros(());over_sum=visual.new_zeros(())
    transitions=0;recons=0;overs=0;branches=[]
    def reconstruction(state,t,valid):
        pred=model.forward_reconstruct(state)
        logits=F.normalize(pred['img_embed'][valid],dim=-1)@pool.T/.05
        return (F.cross_entropy(logits,labels[valid,t],reduction='sum'),
                F.mse_loss(pred['reward'][valid].reshape(-1),dist[valid,t],reduction='sum'))
    for t in range(1,T):
        valid=mask[:,t]
        if not bool(valid.any()): break
        out=model.forward_transit(prev,visual[:,t],text,tm,past_key_values=cache)
        state=torch.cat((out['posterior_stoch'],out['deter']),-1)
        prior_state=torch.cat((out['prior_stoch'],out['deter']),-1)
        k=torch.distributions.kl_divergence(out['posterior_dist'],out['prior_dist']).mean(-1)
        kl_sum+=k[valid].sum();transitions+=int(valid.sum())
        for reconstruction_state in (state,prior_state):
            visual_loss,reward_loss=reconstruction(reconstruction_state,t,valid)
            visual_sum+=visual_loss;reward_sum+=reward_loss;recons+=int(valid.sum())
        new_branches=[]
        for age,hidden,oldcache in branches:
            imagined=model.forward_imagine(hidden,text,tm,past_key_values=oldcache)
            imagined_state=torch.cat((imagined['stoch'],imagined['deter']),-1)
            k=torch.distributions.kl_divergence(out['posterior_dist'],imagined['dist']).mean(-1)
            over_sum+=k[valid].sum();overs+=int(valid.sum())
            visual_loss,reward_loss=reconstruction(imagined_state,t,valid)
            visual_sum+=visual_loss;reward_sum+=reward_loss;recons+=int(valid.sum())
            if age+1<horizon: new_branches.append((age+1,imagined_state.unsqueeze(1),imagined['past_key_values']))
        if horizon>1:new_branches.append((1,prior_state.unsqueeze(1),out['past_key_values']))
        branches=new_branches
        prev=state.unsqueeze(1);cache=out['past_key_values']
    if transitions==0: raise RuntimeError('No transitions in training batch')
    if normalizers is not None:
        transitions, recons, overs = normalizers
    pieces={'kl':kl_sum/transitions,'overshoot_kl':over_sum/max(1,overs),
            'visual_nce':visual_sum/max(1,recons),'reward_mse':reward_sum/max(1,recons)}
    return (kl_weight*pieces['kl']+overshoot_kl_weight*pieces['overshoot_kl']+
            visual_weight*pieces['visual_nce']+reward_weight*pieces['reward_mse']),pieces

def loss_weights(cfg):
    return {k:cfg.get(k,d) for k,d in [('reward_weight',.1),('visual_weight',1.),('kl_weight',2.),('overshoot_kl_weight',2.)]}

class Sampler:
    def __init__(self,size,state=None):
        self.size=size
        self.order=list(np.random.permutation(size).tolist()) if state is None else list(state['order'])
        self.cursor=0 if state is None else state['cursor']
        if sorted(self.order)!=list(range(size)) or not 0<=self.cursor<=size: raise RuntimeError('Invalid sampler state')
    def take(self,n):
        selected=[]
        while len(selected)<n:
            if self.cursor==self.size:
                self.order=np.random.permutation(self.size).tolist();self.cursor=0
            count=min(n-len(selected),self.size-self.cursor)
            selected.extend(self.order[self.cursor:self.cursor+count]);self.cursor+=count
        return selected
    def state(self): return {'order':self.order,'cursor':self.cursor}

class CompactStore(MountedCheckpointStore):
    def __init__(self,local,backup,host_volume,min_free_gib=12,expected_mount=None,expected_host_mount=None,storage_cfg=None,expected_storage_mount=None):
        self.root=Path(backup); self.host_volume=host_volume;self.min_free_gib=min_free_gib
        self.storage_cfg=storage_cfg
        self.storage_mount=verify_storage_mount(storage_cfg,expected_storage_mount) if storage_cfg is not None else None
        self.root.mkdir(exist_ok=True)
        self.mount=expected_mount or backup_mount_identity(self.root,backend='filesystem')
        self.host_mount=verify_host_volume(host_volume,expected_host_mount)
        def check():
            if self.storage_cfg is not None:verify_storage_mount(self.storage_cfg,self.storage_mount)
            validate_backup_root(self.root,backend='filesystem',expected_identity=self.mount)
            verify_host_volume(self.host_volume,self.host_mount)
            space_guard([self.root,self.host_volume],self.min_free_gib)
        super().__init__(local,backup,backup_check=check,keep_local=1,keep_backup=1)
    def save(self,state,head_payload,**kw):
        self.backup_check()
        space_guard([self.root,self.host_volume],self.min_free_gib,
                    additional=2*(tensor_bytes(state)+tensor_bytes(head_payload))+16*1024**2)
        return super().save(state,head_payload,**kw)

def atomic_torch(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def verified_copy(source,destination):
    source,destination=Path(source),Path(destination)
    digest=sha(source)
    if destination.exists():
        if sha(destination)!=digest: raise RuntimeError('Existing immutable backup differs')
        return digest
    tmp=destination.with_name(destination.name+'.partial')
    with source.open('rb') as a,tmp.open('wb') as b:
        shutil.copyfileobj(a,b,8*1024**2);b.flush();os.fsync(b.fileno())
    if sha(tmp)!=digest: raise RuntimeError('Backup readback checksum failed')
    os.replace(tmp,destination)
    if sha(destination)!=digest: raise RuntimeError('Published backup checksum failed')
    return digest

def evaluate(model,data,batch_size,horizon,weights=None):
    was=model.training;model.eval();total=0.;count=0
    try:
        with isolated_rng(917),torch.no_grad():
            for i in range(0,len(data.ids),batch_size):
                ids=list(range(i,min(i+batch_size,len(data.ids))))
                loss,_=wm_loss(model,data.batch(ids,next(model.parameters()).device),horizon,**(weights or {}))
                total+=float(loss)*len(ids);count+=len(ids)
    finally: model.train(was)
    return total/count

def load_selected_weights(run_dir,backup_dir,which='best'):
    """Read SHA-verified artifacts, return raw Glocal state dict + metadata. No new export file."""
    run,backup=Path(run_dir),Path(backup_dir)
    identity=json.loads((run/'identity.json').read_text())
    # Read only; construction of a live writer store is deliberately avoided.
    if which not in ('best','latest'): raise ValueError('Unknown checkpoint selection')
    refs=json.loads((backup/'checkpoints'/'refs.json').read_text())
    pointer=refs[which]
    if not pointer: raise RuntimeError('Requested checkpoint has not been selected')
    ckpt=backup/'checkpoints'/'snapshots'/pointer['checkpoint_id']
    manifest=json.loads((ckpt/'manifest.json').read_text())
    digest=sha(ckpt/'manifest.json')
    if ((ckpt/'COMMITTED').read_text().strip()!=digest or pointer['manifest_sha256']!=digest
            or manifest['run_id']!=refs['run_id'] or pointer['run_id']!=refs['run_id']
            or (which=='best' and not manifest['is_best'])): raise RuntimeError('Invalid checkpoint commit/reference')
    for name,meta in manifest['files'].items():
        if sha(ckpt/name)!=meta['sha256']: raise RuntimeError('Checkpoint checksum failed')
    state=torch.load(ckpt/'state.pt',map_location='cpu',weights_only=True)
    base=backup/'frozen-base.pt'
    if sha(base)!=identity['frozen_base_sha256'] or state['identity_sha256']!=json_sha(identity): raise RuntimeError('Frozen base or identity differs')
    weights=torch.load(base,map_location='cpu',weights_only=True)
    if set(weights)&set(state['trainable_state']): raise RuntimeError('Frozen/trainable overlap')
    weights.update(state['trainable_state'])
    return weights,{'identity':identity,'selected_step':state['step'],'selected_dev_loss':state['best_dev'],'checkpoint':manifest}

def train(cfg,resume=False,stop_after=None):
    configure_runtime(cfg)
    run,backup=Path(cfg['run_dir']),Path(cfg['backup_dir'])
    storage_mount=verify_storage_mount(cfg)
    host_mount=verify_host_volume(cfg['host_volume'])
    validate_cache_contract(cfg)
    if resume:
        if not (run/'identity.json').is_file() or not (backup/'identity.json').is_file(): raise RuntimeError('Strict resume requires both frozen identities')
    elif run.exists() or backup.exists(): raise RuntimeError('Fresh run refuses existing directories')
    else:
        space_guard([Path(cfg['host_volume']),run.parent,backup.parent],cfg['min_free_gib'])
        run.mkdir();backup.mkdir()
    model=make_model(cfg);trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    optimizer=build_optimizer(model,cfg)
    fit=CachedTrajectories(cfg['cache'],'train_fit');dev=CachedTrajectories(cfg['cache'],'train_dev')
    contract={'schema':SCHEMA,'config':cfg,'source_files':code_manifest(),'cache_sha256':fit.manifest['sha256'],
        'cache_inputs':fit.manifest['identity'],
        'base_checkpoint_sha256':sha(cfg['init_checkpoint']),'trainable_names':sorted(trainable),
        'split_manifest_sha256':sha(cfg['split_manifest']),
        'architecture':{'d_model':cfg.get('d_model',512),'layers':cfg.get('layers',2),
            'd_state':cfg.get('d_state',16),'headdim':cfg.get('headdim',64),'dropout':cfg.get('drama_dropout',.1)},
        'model_backend':'drama_mamba2','frozen_encoder_mode':'eval+no_grad FP32 cached inputs',
        'objective':'world model only; GT-next distance MSE (weight .1 default), cross-batch/time InfoNCE; five-step overshoot; reconstruction averaged over all valid posterior/prior/overshoot items (differs from author grouped normalization)',
        'optimizer':optimizer_contract(cfg),'runtime':runtime_identity()}
    if resume:
        identity=json.loads((run/'identity.json').read_text())
        if identity['contract']!=contract or json.loads((backup/'identity.json').read_text())!=identity: raise RuntimeError('Strict resume contract differs')
    else:
        frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k not in trainable}
        space_guard([run,cfg['host_volume']],cfg['min_free_gib'],additional=2*tensor_bytes(frozen)+16*1024**2)
        atomic_torch(run/'frozen-base.pt',frozen);del frozen
        digest=verified_copy(run/'frozen-base.pt',backup/'frozen-base.pt')
        identity={'contract':contract,'frozen_base_sha256':digest,'backup_mount':backup_mount_identity(backup,backend='filesystem'),
                  'host_mount':host_mount,'storage_mount':storage_mount}
        write_json(run/'identity.json',identity);write_json(backup/'identity.json',identity)
    for base in (run/'frozen-base.pt',backup/'frozen-base.pt'):
        if sha(base)!=identity['frozen_base_sha256']: raise RuntimeError('Frozen base checksum mismatch')
    base=torch.load(run/'frozen-base.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(base,strict=False);del base
    store=CompactStore(run/'checkpoints',backup/'checkpoints',cfg['host_volume'],cfg['min_free_gib'],identity['backup_mount'],identity['host_mount'],cfg,identity['storage_mount'])
    step=0;best=float('inf');history=[];sampler=Sampler(len(fit.ids))
    with store.lock():
        if resume:
            state,_,_=store.restore(which='latest')
            if state['identity_sha256']!=json_sha(identity) or set(state['trainable_state'])!=set(trainable): raise RuntimeError('Snapshot identity/parameters mismatch')
            model.load_state_dict(state['trainable_state'],strict=False);optimizer.load_state_dict(state['optimizer'])
            validate_scheduler(state['scheduler'],state['step'],cfg,optimizer)
            step=state['step'];best=state['best_dev'];history=state['history'];sampler=Sampler(len(fit.ids),state['sampler']);set_rng(state['rng'])
        def save(is_best=False):
            state={'schema':SCHEMA,'identity_sha256':json_sha(identity),'step':step,'best_dev':best,'history':history,
                'trainable_state':{n:p.detach().cpu().clone() for n,p in trainable.items()},
                'optimizer':cpu_tree(optimizer.state_dict()),'scheduler':scheduler_state(step,cfg),
                'sampler':sampler.state(),'rng':rng_state()}
            store.save(state,{'kind':'world_model_training_state'},step=step,is_best=is_best)
        if not resume: save()
        model.train();start=time.time()
        while step<cfg['steps']:
            verify_storage_mount(cfg,identity['storage_mount'])
            verify_host_volume(cfg['host_volume'],identity['host_mount'])
            space_guard([run,cfg['host_volume']],cfg['min_free_gib'])
            optimizer.zero_grad(set_to_none=True)
            batch=fit.batch(sampler.take(cfg['batch_size']),next(model.parameters()).device)
            loss,pieces=wm_loss(model,batch,cfg['overshoot_horizon'],**loss_weights(cfg))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable.values(),cfg['grad_clip'],error_if_nonfinite=True)
            lr=scheduled_lr(step+1,cfg)
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step();step+=1
            progress={'phase':'pretraining','step':step,'target':cfg['steps'],'loss':float(loss),'grad_norm':float(norm),
                'lr':lr,
                'seconds':time.time()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                'host_free_gib':shutil.disk_usage(cfg['host_volume']).free/1024**3,
                **{k:float(v.detach()) for k,v in pieces.items()}}
            write_json(run/'progress.json',progress)
            if step%10==0: print('PRETRAIN_PROGRESS',json.dumps(progress),flush=True)
            stopping=stop_after is not None and step>=stop_after
            if step%cfg['eval_every']==0 or step==cfg['steps']:
                score=evaluate(model,dev,cfg['batch_size'],cfg['overshoot_horizon'],loss_weights(cfg))
                history.append({'step':step,'dev_wm_loss':score});is_best=score<best
                if is_best:best=score
                save(is_best)
                write_json(run/'dev-history.json',history)
            elif step%cfg['save_every']==0 or stopping:save()
            if stopping:break
        done=step==cfg['steps']
        write_json(run/'completion.json',{'complete':done,'step':step,'best_dev_wm_loss':best if math.isfinite(best) else None,
            'strict_resume_supported':True,'best_step':min(history,key=lambda x:x['dev_wm_loss'])['step'] if history else None})

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['cache','train'])
    p.add_argument('--config',required=True);p.add_argument('--require-resume',action='store_true')
    p.add_argument('--engineering-limit',type=int);p.add_argument('--stop-after',type=int)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    if os.environ.get('PYTHONHASHSEED')!='0':raise RuntimeError('Launch Python with PYTHONHASHSEED=0')
    if os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('TRANSFORMERS_OFFLINE')!='1':
        raise RuntimeError('Launch with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; prepared assets must be local')
    if cfg.get('backup_backend')!='filesystem':raise RuntimeError('Explicit filesystem backup backend required')
    if a.command=='cache':build_cache(cfg,a.engineering_limit)
    else:train(cfg,a.require_resume,a.stop_after)

if __name__=='__main__':main()
