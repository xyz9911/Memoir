"""No-checkpoint GPU engineering probe; synthetic tensors are not training results."""
import argparse
import json
from pathlib import Path
import time
import torch
from drama_pretrain import make_model, wm_loss

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    model=make_model(cfg);model.train()
    params=[v for v in model.parameters() if v.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4)
    result={'engineering_only':True,'data':'synthetic Gaussian latent features; no navigation metric',
        'total_parameters':sum(v.numel() for v in model.parameters()),
        'trainable_parameters':sum(v.numel() for v in params),'probes':[]}
    frozen={n:v.detach().clone() for n,v in model.named_parameters() if n in (
        'embeddings.token_type_embeddings.weight','img_embeddings.img_linear.weight')}
    for batch,length in ((4,5),(32,15)):
        torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        try:
            data={'language':torch.randn(batch,80,768,device='cuda'),
                  'text_mask':torch.ones(batch,80,dtype=torch.bool,device='cuda'),
                  'visual':torch.randn(batch,length,768,device='cuda'),
                  'mask':torch.ones(batch,length,dtype=torch.bool,device='cuda'),
                  'distance':torch.rand(batch,length,device='cuda')*10}
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                loss,pieces=wm_loss(model,data)
                assert torch.isfinite(loss)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(params,40,error_if_nonfinite=True)
                assert all(v.grad is None for v in model.parameters() if not v.requires_grad)
                opt.step()
            torch.cuda.synchronize()
            for n,v in model.named_parameters():
                if n in frozen: assert torch.equal(v,frozen[n])
            result['probes'].append({'batch':batch,'trajectory_length':length,'updates':2,
                'loss':float(loss),'grad_norm':float(norm),'seconds':time.monotonic()-started,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2})
            del loss,pieces,data
        except torch.cuda.OutOfMemoryError as e:
            result['probes'].append({'batch':batch,'trajectory_length':length,'error':'CUDA out of memory'})
            break
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
