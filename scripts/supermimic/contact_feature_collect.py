"""Frozen, officially launched collection for WM feature ablations. No training.
Snapshot implementation copied unchanged from diagnose_v30_snapshot.py.
Only perturbations and recording differ; solver, reset, DR and health guards stay.
"""
import os,sys,random,runpy,types,json,hashlib
from pathlib import Path
import numpy as np
import torch
import warp as wp
from viserdex.supermimic.simulation import Adapter
from viserdex.supermimic.trainer import Trainer
ROOT=Path(__file__).resolve().parents[2]

class Snapshot:
    """Copy mutable runtime buffers in-place, retaining CUDA graph addresses.

    Includes actuator delay, caches, solver warmstart/contact buffers and RNG.
    This diagnostic snapshot is NOT a supported resumable training checkpoint.
    """
    def __init__(self, roots):
        self.tensors=[];self.arrays=[];self.objects=[];self.containers=[];seen=set()
        def visit(x):
            if id(x) in seen:return
            seen.add(id(x))
            if isinstance(x,torch.Tensor):
                self.tensors.append((x,x.clone()));return
            if isinstance(x,wp.array):
                self.arrays.append((x,wp.clone(x)));return
            if isinstance(x,dict):
                self.containers.append((x,x.copy()))
                for v in x.values():visit(v)
            elif isinstance(x,list):
                self.containers.append((x,x.copy()))
                for v in x:visit(v)
            elif isinstance(x,tuple):
                for v in x:visit(v)
            elif (not isinstance(x,(type,types.ModuleType)) and hasattr(x,'__dict__')
                  and type(x).__module__.startswith(('isaaclab','viserdex','newton','mujoco_warp'))):
                values=vars(x).copy();self.objects.append((x,values))
                for v in values.values():visit(v)
        for root in roots:visit(root)
        self.rng=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        wp.synchronize()
    def restore(self):
        for obj,values in self.objects:
            obj.__dict__.clear();obj.__dict__.update(values)
        for obj,values in self.containers:
            if isinstance(obj,dict):obj.clear();obj.update(values)
            else:obj[:]=values
        for a,b in self.tensors:
            index=tuple(0 if stride==0 else slice(None) for stride in a.stride())
            a[index].copy_(b[index])
        for a,b in self.arrays:wp.copy(a,b)
        random.setstate(self.rng[0]);np.random.set_state(self.rng[1]);torch.set_rng_state(self.rng[2]);torch.cuda.set_rng_state_all(self.rng[3])
        wp.synchronize()

if not all(arg in sys.argv for arg in ('--evaluate','--resume','--env_config')):
    raise ValueError('Read-only collector requires evaluate/resume/env_config')
if any(arg in sys.argv for arg in ('--smoke','--calibrate_termination','--check_joint_update')):
    raise ValueError('Collector cannot update models')
out=Path(os.environ['CONTACT_DATA_OUTPUT']);out.mkdir(parents=True,exist_ok=False)
seed=int(sys.argv[sys.argv.index('--seed')+1])
probe_steps=[int(x) for x in os.environ['CONTACT_DATA_STEPS'].split(',')]
checkpoint=Path(sys.argv[sys.argv.index('--resume')+1])
manifest=dict(protocol='contact_features_branch_v1',seed=seed,steps=probe_steps,
 checkpoint=str(checkpoint),status='running',snapshot_reports=[],
 scope='Frozen deterministic policy, then open-loop pulses/holds. Original physical and sensor randomization retained.')
trainer_ref=None
excitation=None
original_restore=Trainer.restore

def model_digest(trainer):
    h=hashlib.sha256()
    for name in ('wm','actor','critic','target_critic'):
        for key,v in getattr(trainer,name).state_dict().items():
            h.update((name+'.'+key).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def restore(self,*args,**kwargs):
    global trainer_ref
    original_restore(self,*args,**kwargs);trainer_ref=self
    manifest['model_hash_before']=model_digest(self)
    manifest['optimizer_steps_before']=dict(self.optimizer_steps)
Trainer.restore=restore
original_step=Adapter.step
count=0
FIELDS=('state','next_physical','action','done','terminated','reward')

def step(self,action):
    global count, excitation
    if os.environ.get('CONTACT_HOME_EXCITATION') == '1':
        from viserdex.supermimic.warmup_excitation import HomeExcitation
        if excitation is None:
            excitation=HomeExcitation(self.limits,trainer_ref.cfg)
            manifest['collection_driver']='bounded_HOME_excitation; fixed full envelope; no Actor updates'
        action=excitation.action(0)
    if count in probe_steps:
        from isaaclab_newton.physics import NewtonManager as N
        snap=Snapshot([self,N.backend,N._solver,N._contacts,N._collision_pipeline])
        start=self.state.clone();history=self.history.clone()
        idx=torch.arange(len(action),device=action.device)
        snap_index=probe_steps.index(count)
        direction=(idx+5*snap_index+seed)%21
        amplitudes=action.new_tensor([.5,1.,2.])[(idx//21+snap_index)%3]
        hold_steps=1+4*((idx//7+snap_index)%2)
        half=(self.limits[:,:,1]-self.limits[:,:,0])/2
        delta=amplitudes*torch.pi/180/half[idx,direction]
        eligible=action[idx,direction].abs()+delta<1
        folder=out/str(count);folder.mkdir()
        report=dict(source_step=count,worlds=len(action),eligible=int(eligible.sum()),
                    snapshot_tensors=len(snap.tensors),snapshot_warp_arrays=len(snap.arrays),
                    branches={},contact_peaks={})
        for label,sign in [('base',0),('repeat',0),('repeat2',0),('repeat3',0),('plus',1),('minus',-1),('plus2',1),('minus2',-1)]:
            snap.restore()
            assert torch.equal(self.state,start) and torch.equal(self.history,history)
            rows=[]
            for k in range(8):
                command=action.clone()
                command[idx,direction]+=sign*delta*eligible*(k<hold_steps)
                record=original_step(self,command)[0]
                rows.append({key:record[key].detach().cpu().clone() for key in FIELDS})
                for key,value in self.contact_peaks.items():report['contact_peaks'][key]=max(report['contact_peaks'].get(key,0),value)
            target=folder/(label+'.pt')
            torch.save(dict(records=rows,context=self.context.detach().cpu().clone(),
                limits=self.limits.detach().cpu().clone(),seed=seed,source_step=count,sign=sign,
                eligible=eligible.cpu(),joint=direction.cpu(),amplitudes_deg=amplitudes.cpu(),
                hold_steps=hold_steps.cpu()),target)
            report['branches'][label]=dict(sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                failed_worlds=int(torch.stack([x['terminated'] for x in rows]).any(0).sum()))
        snap.restore();assert torch.equal(self.state,start) and torch.equal(self.history,history)
        (folder/'manifest.json').write_text(json.dumps(report,indent=2))
        manifest['snapshot_reports'].append(report)
        print('CONTACT_FEATURE_SNAPSHOT',count,report['eligible'],flush=True)
    count+=1
    result=original_step(self,action)
    if excitation is not None:excitation.reset(result[0]['done'])
    return result
Adapter.step=step
try:
    runpy.run_path(str(ROOT/'scripts/supermimic/train.py'),run_name='__main__')
    assert len(manifest['snapshot_reports'])==len(probe_steps)
    manifest['model_hash_after']=model_digest(trainer_ref)
    manifest['optimizer_steps_after']=dict(trainer_ref.optimizer_steps)
    assert manifest['model_hash_before']==manifest['model_hash_after']
    assert manifest['optimizer_steps_before']==manifest['optimizer_steps_after']
    manifest['status']='complete'
finally:
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
