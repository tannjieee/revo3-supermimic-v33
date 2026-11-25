"""Controlled, offline WM-only 2x2 feature ablation on frozen real branches.

No simulator launch, Actor/Critic updates, production Config changes, or resumable
policy exports. Four arms share padded architecture and function-preserving init.
New feature statistics use training real transitions only. Test is opened after
all fixed-budget fits finish. This tests proxy geometry, not full ContactNets.
"""
from pathlib import Path
import argparse, copy, hashlib, itertools, json, math, sys, time
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/viserdex'))
from viserdex.supermimic.core import Config, action_target, advance, physical_delta, integrate_physical
from viserdex.supermimic.networks import Dynamics, RunningNorm
from viserdex.supermimic.geometry import HandObjectGeometry
from viserdex.supermimic.policy_contract import absolute_joint_target
from viserdex.supermimic.state_layout import COMPACT
from viserdex.supermimic.trainer import Trainer
from viserdex.distillation.config import PerceptionNoiseCfg
BRANCHES=('base','repeat','repeat2','repeat3','plus','minus','plus2','minus2')
TRAIN_BRANCHES=('base','plus','minus')
GROUPS=('q','qd','position','rotation','velocity','omega')
SIZES=(21,21,3,3,3,3)
FLOORS=(1e-7,1e-6,1e-7,1e-6,1e-6,1e-6)

def digest_state(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

class FeatureDynamics(Dynamics):
    def __init__(self,payload,arm):
        cfg=Config(**payload['config']);assert not cfg.geometry_enabled and cfg.amp=='none'
        super().__init__(len(payload['wm']['context_scale']),cfg)
        self.load_state_dict(payload['wm'],strict=True)
        self.feature_geometry=HandObjectGeometry(payload['geometry_spec'])
        self.feature_norm=RunningNorm(self.feature_geometry.feature_dim+105)
        self.feature_projection=nn.Linear(len(self.feature_norm.mean),cfg.wm_hidden[0],bias=False)
        nn.init.zeros_(self.feature_projection.weight)
        self.register_buffer('feature_mask',torch.tensor([float(arm in ('geometry','both'))]*self.feature_geometry.feature_dim+[float(arm in ('actuator','both'))]*105))
        self.register_buffer('limits',torch.tensor(cfg.network_joint_limits,dtype=torch.float32))
        self.register_buffer('encoder_bias',torch.tensor(cfg.encoder_bias_rad,dtype=torch.float32))
        metadata=payload['metadata'];self.kp_slice=slice(*metadata['context_schema']['actuator.fingers.kp']);self.kd_slice=slice(*metadata['context_schema']['actuator.fingers.kd'])
        info=metadata['identified_hand'];self.delay_fraction=info['effective_delay_s']/cfg.control_dt
        assert 0<self.delay_fraction<1
        override=info.get('actuator_output_limit_override')
        effort=([override['effort_limit_Nm']]*21 if override else metadata['context_manifest']['identified_fixed']['effective_effort_limit_Nm'])
        self.register_buffer('effort_limit',torch.tensor(effort,dtype=torch.float32))
        self.arm=arm

    def actuator_features(self,s,target,context):
        # Targets are encoder coordinates; q is geometric. No target filtering.
        current_error=target+self.encoder_bias-s[:,:21]
        previous_target=absolute_joint_target(s[:,COMPACT.last_action],self.limits)
        old_error=previous_target+self.encoder_bias-s[:,:21]
        ready=s[:,COMPACT.delay_ready]
        previous_error=torch.where(ready.bool(),old_error,current_error)
        kp,kd=context[:,self.kp_slice],context[:,self.kd_slice]
        current_torque=(kp*current_error-kd*s[:,21:42]).clamp(-self.effort_limit,self.effort_limit)/self.effort_limit
        previous_torque=(kp*previous_error-kd*s[:,21:42]).clamp(-self.effort_limit,self.effort_limit)/self.effort_limit
        # This frozen-start weighted torque is a feature, NOT exact PD integration.
        delay=ready*self.delay_fraction
        weighted=(1-delay)*current_torque+delay*previous_torque
        half=(self.limits[:,1]-self.limits[:,0])/2
        return torch.cat((current_error/half,previous_error/half,current_torque,previous_torque,weighted),-1)

    def extra_features(self,s,target,context):
        return torch.cat((self.feature_geometry.features(s),self.actuator_features(s,target,context)),-1)

    def forward(self,s,target,context):
        base=self.input_norm(self.inputs(s,target,context))
        features=self.feature_norm(self.extra_features(s,target,context))*self.feature_mask
        x=self.net[0](base)+self.feature_projection(features)
        for layer in self.net[1:]:x=layer(x)
        return integrate_physical(s,x*self.delta_norm.std+self.delta_norm.mean)


def dataset(paths,device):
    parts=[]; metadata=[]
    for root in paths:
        root=Path(root);manifest=json.loads((root/'manifest.json').read_text());assert manifest['status']=='complete'
        assert manifest['model_hash_before']==manifest['model_hash_after']
        for folder in sorted((p.parent for p in root.glob('*/base.pt')),key=lambda p:int(p.name)):
            m=json.loads((folder/'manifest.json').read_text())
            raw={}
            for b in BRANCHES:
                path=folder/(b+'.pt');assert sha(path)==m['branches'][b]['sha256']
                raw[b]=torch.load(path,map_location='cpu',weights_only=False)
            ref=raw['base']
            for b in raw.values():
                assert torch.equal(b['records'][0]['state'],ref['records'][0]['state'])
                assert torch.equal(b['context'],ref['context']) and torch.equal(b['limits'],ref['limits'])
                assert len(b['records'])==8
            n=len(ref['context'])
            data={b:{k:torch.stack([row[k] for row in d['records']]) for k in ('state','next_physical','action','done','terminated','reward')} for b,d in raw.items()}
            # Verify hold timing, perturbation signs, amplitudes and unchanged baseline sequence.
            for sign,b in ((1,'plus'),(-1,'minus')):
                expected=torch.zeros_like(data[b]['action']);idx=torch.arange(n);j=ref['joint'];half=(ref['limits'][:,:,1]-ref['limits'][:,:,0])/2
                amp=ref['amplitudes_deg']*math.pi/180/half[idx,j]
                for t in range(8):expected[t,idx,j]=sign*amp*ref['eligible']*(t<ref['hold_steps'])
                torch.testing.assert_close(data[b]['action']-data['base']['action'],expected,atol=1.2e-7,rtol=1e-5)
            parts.append(dict(data=data,context=ref['context'],limits=ref['limits'],eligible=ref['eligible'],joint=ref['joint'],amplitude=ref['amplitudes_deg'],hold=ref['hold_steps'],cluster=ref['seed']*100000+torch.arange(n),source_step=torch.full((n,),ref['source_step'])))
            metadata.append(dict(path=str(folder),seed=ref['seed'],worlds=n,source_step=ref['source_step'],manifest_sha256=sha(folder/'manifest.json')))
    out={'data':{b:{k:torch.cat([p['data'][b][k] for p in parts],1).to(device) for k in parts[0]['data'][b]} for b in BRANCHES}}
    for k in ('context','limits','eligible','joint','amplitude','hold','cluster','source_step'):out[k]=torch.cat([p[k] for p in parts]).to(device)
    out['metadata']=metadata
    valid=out['eligible'][None].expand(8,-1).clone()
    for b in BRANCHES:valid &= (~out['data'][b]['done']).cumprod(0).bool()
    out['valid']=valid
    out['response']=(physical_delta(out['data']['minus']['next_physical'],out['data']['plus']['next_physical'])+physical_delta(out['data']['minus2']['next_physical'],out['data']['plus2']['next_physical']))/2
    noise=out['response'].new_zeros(8,len(out['context']),6)
    pairs=list(itertools.combinations(('base','repeat','repeat2','repeat3'),2))+[('plus','plus2'),('minus','minus2')]
    for a,b in pairs:
        difference=physical_delta(out['data'][a]['next_physical'],out['data'][b]['next_physical'])
        noise=torch.maximum(noise,torch.stack([v.norm(dim=-1) for v in difference.split(SIZES,-1)],-1))
    out['noise']=noise
    out['resolved']=valid[...,None] & (torch.stack([v.norm(dim=-1) for v in out['response'].split(SIZES,-1)],-1)>10*noise.clamp_min(noise.new_tensor(FLOORS)))
    return out


def window(d,ids):
    w={k:torch.cat([d['data'][b][k][:,ids].transpose(0,1) for b in TRAIN_BRANCHES]) for k in d['data']['base']}
    w['env_id']=ids.repeat(3)
    w['mask']=torch.cat((torch.ones_like(w['done'][:,:1]),(~w['done'][:,:-1]).cumprod(1).bool()),1).float()
    return w

@torch.no_grad()
def fit_feature_statistics(model,d,cfg):
    # Three distinct action branches, excluding all post-reset suffixes.
    for b in TRAIN_BRANCHES:
        row=d['data'][b]
        live=torch.cat((torch.ones_like(row['done'][:1]),(~row['done'][:-1]).cumprod(0).bool()),0)
        for t in range(8):
            ids=live[t].nonzero().flatten()
            # Tensor.split returns one empty chunk for an empty tensor. Updating
            # RunningNorm with that chunk would poison its mean/variance.
            if ids.numel()==0:
                continue
            for batch in ids.split(256):
                s=row['state'][t,batch];target=action_target(s,row['action'][t,batch],d['limits'][batch],cfg)
                features=model.extra_features(s,target,d['context'][batch]);assert torch.isfinite(features).all()
                model.feature_norm.update(features)
    assert all(torch.isfinite(value).all() for value in model.feature_norm.state_dict().values())

@torch.no_grad()
def rollout(model,d,cfg):
    result={b:[] for b in TRAIN_BRANCHES}
    for ids in torch.arange(len(d['context']),device=d['context'].device).split(128):
        s=torch.cat([d['data'][b]['state'][0,ids] for b in TRAIN_BRANCHES]);ctx=d['context'][ids].repeat(3,1);limits=d['limits'][ids].repeat(3,1,1)
        seq=[]
        for t in range(8):
            a=torch.cat([d['data'][b]['action'][t,ids] for b in TRAIN_BRANCHES]);target=action_target(s,a,limits,cfg)
            pred=model(s,target,ctx);assert torch.isfinite(pred).all()
            seq.append(pred);s=advance(s,pred,a,target,s[:,COMPACT.goal_rot],500,hold_cfg=cfg)
        seq=torch.stack(seq)
        for b,values in zip(TRAIN_BRANCHES,seq.split(len(ids),1)):result[b].append(values.cpu())
    return {b:torch.cat(chunks,1) for b,chunks in result.items()}


def qstats(x):
    if x.numel()==0:return None
    return dict(zip(('p10','median','p90'),torch.quantile(x.float(),torch.tensor([.1,.5,.9])).tolist()))

def score(pred,d):
    real=d['response'].cpu();estimated=physical_delta(pred['minus'],pred['plus']);valid=d['valid'].cpu();resolved=d['resolved'].cpu();noise=d['noise'].cpu()
    output={};base_err=physical_delta(d['data']['base']['next_physical'].cpu(),pred['base'])
    persistence=physical_delta(d['data']['base']['next_physical'].cpu(),d['data']['base']['state'][0,:,:55].cpu()[None].expand(8,-1,-1))
    masks={'all':torch.ones(len(d['context']),dtype=torch.bool),'pulse':d['hold'].cpu()==1,'hold5':d['hold'].cpu()==5}
    for amp in (.5,1.,2.):masks[f'amp{amp:g}']=d['amplitude'].cpu()==amp
    for name,subset in masks.items():
        view={}
        for t in (0,1,3,7):
            groups={}
            for g,(r,m,be,pe) in enumerate(zip(real[t].split(SIZES,-1),estimated[t].split(SIZES,-1),base_err[t].split(SIZES,-1),persistence[t].split(SIZES,-1))):
                mask=resolved[t,:,g]&subset;live=valid[t]&subset
                rn,mn=r.norm(dim=-1),m.norm(dim=-1)
                values=dict(valid=int(live.sum()),resolved=int(mask.sum()),real_norm=qstats(rn[live]),noise_norm=qstats(noise[t,live,g]),base_rmse=float(be[live].square().mean().sqrt()) if live.any() else None,persistence_rmse=float(pe[live].square().mean().sqrt()) if live.any() else None)
                if mask.any():
                    rr,mm=r[mask],m[mask];cos=(rr*mm).sum(-1)/(rr.norm(dim=-1)*mm.norm(dim=-1)).clamp_min(1e-20)
                    values.update(cosine_mean=float(cos.mean()),positive_fraction=float((cos>0).float().mean()),amplitude_ratio=qstats(mn[mask]/rn[mask]),relative_vector_error=float((mm-rr).norm()/rr.norm()),per_sample_relative_error=qstats((mm-rr).norm(dim=-1)/rn[mask]),zero_response_baseline_relative_error=1.)
                if g==3:
                    quiet=live&(rn<=1e-5)&(noise[t,:,g]<=1e-6)
                    values['quiet']=dict(count=int(quiet.sum()),false_response_gt1e_4=int((mn[quiet]>1e-4).sum()),model_norm=qstats(mn[quiet]))
                groups[GROUPS[g]]=values
            view[str(t+1)]=groups
        output[name]=view
    return output


def assert_frozen(trainer,payload,frozen_hashes,feature_stats):
    for name,h in frozen_hashes.items():assert digest_state(getattr(trainer,name))==h,name
    for prefix in ('input_norm.','delta_norm.'):
        for k,v in payload['wm'].items():
            if k.startswith(prefix):assert torch.equal(trainer.wm.state_dict()[k].cpu(),v.cpu()),k
    for k,v in feature_stats.items():assert torch.equal(trainer.wm.feature_norm.state_dict()[k].cpu(),v.cpu()),k


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--selfcheck-only',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    plan=json.loads((a.experiment/'plan.json').read_text());payload=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)
    cfg=Config(**payload['config']);device='cuda:0'
    # Build from the approved frozen checkpoint; only WM replacement below varies.
    train=dataset([a.experiment/f'branches_seed{s}' for s in plan['train_seeds']],device)
    val=dataset([a.experiment/f'branches_seed{s}' for s in plan['validation_seeds']],device)
    assert set(train['cluster'].cpu().tolist()).isdisjoint(val['cluster'].cpu().tolist())
    unit=FeatureDynamics(payload,'both').to(device);fit_feature_statistics(unit,train,cfg)
    stats=copy.deepcopy(unit.feature_norm.state_dict())
    ids=torch.arange(32,device=device);s=train['data']['base']['state'][0,ids];ctx=train['context'][ids];target=action_target(s,train['data']['base']['action'][0,ids],train['limits'][ids],cfg)
    original=Dynamics(ctx.shape[-1],cfg).to(device);original.load_state_dict(payload['wm'])
    with torch.no_grad():
        error=(unit(s,target,ctx)-original(s,target,ctx)).abs().max().item();assert error==0,error
        reset=s.clone();reset[:,COMPACT.delay_ready]=0
        f=unit.actuator_features(reset,target,ctx).split(21,-1)
        assert torch.equal(f[0],f[1]) and torch.equal(f[2],f[3]) and torch.equal(f[2],f[4])
        # Independent bias/sign/limit oracle for the current torque feature.
        oracle=(ctx[:,23:44]*(target+torch.tensor(cfg.encoder_bias_rad,device=device)-s[:,:21])-ctx[:,44:65]*s[:,21:42]).clamp(-1.28,1.28)/1.28
        pd_error=(unit.actuator_features(s,target,ctx)[:,42:63]-oracle).abs().max().item()
        torch.testing.assert_close(unit.actuator_features(s,target,ctx)[:,42:63],oracle,atol=1e-7,rtol=2e-7)
    a.output.mkdir(parents=True,exist_ok=False)
    save_json(a.output/'selfcheck.json',dict(initial_prediction_max_error=error,empty_delay_uses_current_target=True,pd_bias_sign_limit_oracle=True,pd_oracle_max_error=pd_error,train_worlds=int(train['cluster'].unique().numel()),val_worlds=int(val['cluster'].unique().numel()),train_state_world_starts=len(train['context']),val_state_world_starts=len(val['context']),source_checkpoint_sha256=sha(plan['checkpoint'])))
    if a.selfcheck_only:return
    torch.save({k:v.cpu() for k,v in stats.items()},a.output/'feature_statistics_train_only.pt')
    inv=torch.zeros(len(train['context']),4,device=device);inv[:,3]=1
    t=Trainer(cfg,PerceptionNoiseCfg(**payload['noise_config']),train['context'],train['limits'],inv,device,episode_steps=500,geometry_spec=payload['geometry_spec'])
    for name in ('actor','critic','target_critic'):
        getattr(t,name).load_state_dict(payload[name]);getattr(t,name).requires_grad_(False)
    frozen_hashes={name:digest_state(getattr(t,name)) for name in ('actor','critic','target_critic')}
    report=dict(status='fitting',plan=plan,script_sha256=sha(__file__),train_metadata=train['metadata'],validation_metadata=val['metadata'],normalizers_updated=False,new_feature_statistics_training_only=True,actor_critic_hashes=frozen_hashes,initial_validation=score(rollout(unit,val,cfg),val),fits={})
    save_json(a.output/'result.json',report)
    del original,unit
    saved=[]
    for sampling_seed in plan['optimizer_seeds']:
        generator=torch.Generator().manual_seed(sampling_seed)
        batches=torch.randint(len(train['context']),(plan['updates'],plan['batch_state_world_starts']),generator=generator)
        for arm in plan['arms']:
            key=f'{arm}_seed{sampling_seed}';torch.manual_seed(sampling_seed)
            t.wm=FeatureDynamics(payload,arm).to(device);t.wm.feature_norm.load_state_dict(stats)
            t.cfg.wm_batch=plan['batch_state_world_starts']*3
            optim=torch.optim.Adam(t.wm.parameters(),lr=plan['learning_rate']);entry=dict(arm=arm,sampling_seed=sampling_seed,trace=[],validation={},parameter_count=sum(p.numel() for p in t.wm.parameters()))
            torch.cuda.synchronize();start=time.monotonic()
            for i,indices in enumerate(batches):
                w=window(train,indices.to(device));optim.zero_grad(set_to_none=True)
                loss,_,phys,_,task=t._wm_loss(w);loss=loss/8
                assert torch.isfinite(loss),key
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(t.wm.parameters(),1.,error_if_nonfinite=True);optim.step()
                if (i+1)%25==0:
                    entry['trace'].append(dict(update=i+1,loss=float(loss.detach()),unclipped_gradient_norm=float(grad),physical=(phys/8).tolist(),task=(task/8).tolist()))
                    print(key,entry['trace'][-1],flush=True)
                if i+1 in (50,100,200):
                    entry['validation'][str(i+1)]=score(rollout(t.wm,val,cfg),val)
            torch.cuda.synchronize();entry['seconds_including_validation']=time.monotonic()-start
            assert_frozen(t,payload,frozen_hashes,stats)
            weight_path=a.output/(key+'_wm_only.pt')
            torch.save(dict(experimental_only=True,resumable_policy=False,arm=arm,sampling_seed=sampling_seed,updates=plan['updates'],source_checkpoint=plan['checkpoint'],wm={k:v.cpu() for k,v in t.wm.state_dict().items()}),weight_path)
            entry['weights_sha256']=sha(weight_path);report['fits'][key]=entry;saved.append((key,weight_path))
            save_json(a.output/'result.json',report)
            print('FIT_COMPLETE',key,entry['seconds_including_validation'],flush=True)
    # Never use test observations, statistics or scores to select a model/update.
    report['status']='all_fits_complete_before_test';save_json(a.output/'result.json',report)
    test=dataset([a.experiment/f'branches_seed{s}' for s in plan['test_seeds']],device)
    assert set(test['cluster'].cpu().tolist()).isdisjoint(torch.cat((train['cluster'],val['cluster'])).cpu().tolist())
    report['test_metadata']=test['metadata']
    initial=FeatureDynamics(payload,'baseline').to(device);initial.feature_norm.load_state_dict(stats)
    predictions=rollout(initial,test,cfg);report['initial_test']=score(predictions,test)
    torch.save(predictions,a.output/'initial_test_predictions.pt');del initial
    for key,path in saved:
        fitted=torch.load(path,map_location='cpu',weights_only=False);t.wm=FeatureDynamics(payload,fitted['arm']).to(device);t.wm.load_state_dict(fitted['wm'])
        predictions=rollout(t.wm,test,cfg);report['fits'][key]['test']=score(predictions,test)
        torch.save(predictions,a.output/(key+'_test_predictions.pt'))
        assert_frozen(t,payload,frozen_hashes,stats)
    report['status']='complete';report['final_frozen_hashes']={name:digest_state(getattr(t,name)) for name in frozen_hashes};save_json(a.output/'result.json',report)
    print('EXPERIMENT_COMPLETE',a.output,flush=True)
if __name__=='__main__':main()
