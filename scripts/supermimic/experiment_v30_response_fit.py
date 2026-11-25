"""Bounded offline WM-only ablation. No Actor/Critic or simulator updates.

Fit ordinary production WM losses vs the same losses plus paired response
supervision. The first 48 worlds in each 64-world snapshot train; the supplied
independent test seed is evaluation only. Repeated snapshots are correlated.
The saved output is WM-only experimental weights, not a resumable policy.
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'source/viserdex'))
from viserdex.supermimic.core import Config, action_target, advance, rotation_log, quat_mul, quat_conjugate
from viserdex.supermimic.trainer import Trainer
from viserdex.supermimic.state_layout import COMPACT
from viserdex.distillation.config import PerceptionNoiseCfg


def rotation(a,b):
    return rotation_log(quat_mul(a[...,45:49],quat_conjugate(b[...,45:49])))


def load(path,device):
    if not (path/'base.pt').exists():
        parts=[load(p.parent,device) for p in sorted(path.glob('*/base.pt'))]
        if not parts:raise ValueError(f'No branch snapshots in {path}')
        return merge(parts)
    keys=('base','repeat','repeat2','repeat3','plus','minus','plus2','minus2')
    raw={k:torch.load(path/(k+'.pt'),map_location='cpu',weights_only=False) for k in keys}
    for branch in raw.values():
        torch.testing.assert_close(branch['records'][0]['state'],raw['base']['records'][0]['state'],rtol=0,atol=0)
    data={k:{field:torch.stack([r[field] for r in b['records']]).to(device) for field in ('state','next_physical','action','done','terminated','reward')} for k,b in raw.items()}
    ctx=raw['base']['context'].to(device);limits=raw['base']['limits'].to(device)
    eligible=(data['plus']['action'][0]-data['minus']['action'][0]).abs().amax(-1)>0
    valid=eligible[None].expand(8,-1).clone()
    for b in data.values(): valid &= (~b['done']).cumprod(0).bool()
    response=(rotation(data['plus']['next_physical'],data['minus']['next_physical'])+rotation(data['plus2']['next_physical'],data['minus2']['next_physical']))/2
    noise=torch.zeros_like(response[...,0])
    for a,b in [('repeat','base'),('repeat2','base'),('repeat3','base'),('plus','plus2'),('minus','minus2')]:
        noise=torch.maximum(noise,rotation(data[a]['next_physical'],data[b]['next_physical']).norm(dim=-1))
    resolved=valid & (response.norm(dim=-1)>10*noise.clamp_min(1e-6))
    return dict(data=data,context=ctx,limits=limits,response=response,resolved=resolved)


def merge(parts):
    return dict(data={k:{field:torch.cat([p['data'][k][field] for p in parts],dim=1) for field in parts[0]['data'][k]} for k in parts[0]['data']},
        context=torch.cat([p['context'] for p in parts]),limits=torch.cat([p['limits'] for p in parts]),
        response=torch.cat([p['response'] for p in parts],dim=1),resolved=torch.cat([p['resolved'] for p in parts],dim=1))


def window(dataset,ids):
    rows=dataset['data']; w={}
    for field in ('state','next_physical','action','done','terminated','reward'):
        w[field]=torch.cat([rows[k][field][:,ids].transpose(0,1) for k in ('base','plus','minus')])
    w['env_id']=ids.repeat(3)
    w['mask']=torch.cat((torch.ones_like(w['done'][:,:1]),(~w['done'][:,:-1]).cumprod(1).bool()),1).float()
    return w


def rollout(trainer,dataset,ids):
    states={k:dataset['data'][k]['state'][0,ids] for k in ('base','plus','minus')};preds={k:[] for k in states}
    for step in range(8):
        for k in states:
            action=dataset['data'][k]['action'][step,ids];s=states[k]
            target=action_target(s,action,dataset['limits'][ids],trainer.cfg)
            nxt=trainer.wm(s,target,dataset['context'][ids]);preds[k].append(nxt)
            states[k]=advance(s,nxt,action,target,s[:,COMPACT.goal_rot],500,hold_cfg=trainer.cfg)
    return {k:torch.stack(v) for k,v in preds.items()}


def response_loss(trainer,dataset,ids,mode):
    pred=rollout(trainer,dataset,ids)
    estimated=rotation(pred['plus'],pred['minus']);real=dataset['response'][:,ids]
    mask=dataset['resolved'][:,ids]
    # Per-response relative vector loss. Never supervise noise-dominated pairs.
    if mode=='paired_pooled':
        # One training-only scale: do not overweight each nearly stationary pair.
        scale=real.norm(dim=-1)[mask].median().clamp_min(1e-4)
        return torch.nn.functional.smooth_l1_loss(estimated[mask]/scale,real[mask]/scale)
    error=(estimated-real).square().sum(-1)/real.square().sum(-1).clamp_min(1e-10)
    return error[mask].mean()


@torch.no_grad()
def evaluate(trainer,dataset,ids):
    pred=rollout(trainer,dataset,ids); r=dataset['response'][:,ids];m=rotation(pred['plus'],pred['minus']);result={}
    for step in (0,7):
        mask=dataset['resolved'][step,ids];rr=r[step,mask];mm=m[step,mask]
        denom=(rr.norm(dim=-1)*mm.norm(dim=-1)).clamp_min(1e-20)
        result[str(step+1)]={'samples':int(mask.sum()),'cosine':((rr*mm).sum(-1)/denom).mean().item(),'relative_error':((mm-rr).norm()/rr.norm().clamp_min(1e-20)).item(),'base_orientation_error_deg':rotation(pred['base'][step],dataset['data']['base']['next_physical'][step,ids]).norm(dim=-1).mean().item()*180/torch.pi}
    old=trainer.context,trainer.limits,trainer.cfg.wm_batch
    trainer.context,trainer.limits,trainer.cfg.wm_batch=dataset['context'],dataset['limits'],3*len(ids)
    loss,_,physical,_,task=trainer._wm_loss(window(dataset,ids))
    trainer.context,trainer.limits,trainer.cfg.wm_batch=old
    result['production_loss']=loss.item()/8;result['physical_components']=physical.tolist();result['task_components']=task.tolist()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True);parser.add_argument('--train',type=Path,nargs='+',required=True);parser.add_argument('--test',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--updates',type=int,default=100)
    parser.add_argument('--variants',nargs='+',choices=('ordinary','paired_relative','paired_pooled'),default=['ordinary','paired_relative','paired_pooled'])
    args=parser.parse_args();assert 1<=args.updates<=200
    def files(path):return {path.resolve()/'base.pt'} if (path/'base.pt').exists() else {p.resolve() for p in path.glob('*/base.pt')}
    if set.union(*(files(path) for path in args.train)) & files(args.test):
        raise ValueError('Training and evaluation snapshots must be disjoint')
    args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(20261105)
    device='cuda:0';payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=Config(**payload['config']);cfg.wm_batch=144
    train,test=merge([load(path,device) for path in args.train]),load(args.test,device)
    inv=torch.zeros(len(train['context']),4,device=device);inv[:,3]=1
    t=Trainer(cfg,PerceptionNoiseCfg(**payload['noise_config']),train['context'],train['limits'],inv,device,episode_steps=500)
    t.actor.load_state_dict(payload['actor']);t.actor.requires_grad_(False)
    ids=torch.arange(len(train['context']),device=device);ids=ids[ids%64<48]
    testids=torch.arange(len(test['context']),device=device);cfg.wm_batch=3*len(ids);w=window(train,ids)
    report=dict(checkpoint=str(args.checkpoint),train=[str(p) for p in args.train],test=str(args.test),updates=args.updates,learning_rate=1e-5,training_state_world_pairs=len(ids),test_state_world_pairs=len(testids),actor_updated=False,critic_updated=False,normalizers_updated=False,variants={})
    t.wm.load_state_dict(payload['wm']);report['before']={'train':evaluate(t,train,ids),'test':evaluate(t,test,testids)}
    for label in args.variants:
        weight=0. if label=='ordinary' else .1
        t.wm.load_state_dict(payload['wm']);optimizer=torch.optim.Adam(t.wm.parameters(),lr=1e-5)
        trace=[]
        for update in range(args.updates):
            optimizer.zero_grad(set_to_none=True)
            ordinary=t._wm_loss(w)[0]/8
            paired=response_loss(t,train,ids,label) if weight else ordinary*0
            loss=ordinary+weight*paired
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite experimental loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(t.wm.parameters(),1.);optimizer.step()
            if (update+1)%20==0:
                trace.append(dict(update=update+1,ordinary=ordinary.item(),paired=paired.item()));print(label,trace[-1],flush=True)
        report['variants'][label]=dict(pair_weight=weight,trace=trace,train=evaluate(t,train,ids),test=evaluate(t,test,testids))
        for key,value in t.actor.state_dict().items():
            assert torch.equal(value.cpu(),payload['actor'][key]),key
        for key,value in t.wm.state_dict().items():
            if key.startswith(('input_norm.','delta_norm.')):
                assert torch.equal(value.cpu(),payload['wm'][key]),key
        torch.save({'wm':t.wm.state_dict(),'experimental_only':True,'source_checkpoint':str(args.checkpoint)},args.output/(label+'_wm_only.pt'))
        (args.output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
