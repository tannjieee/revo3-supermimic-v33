"""Diagnostic-only in-process branch/restore. Zero-branch controls quantify residual solver nondeterminism.
Uses the unchanged train entrypoint, solver, randomized environment and guards.
No optimizer updates; invoke with --evaluate --eval_steps 129.
"""
import os, random, runpy, types, sys
from pathlib import Path
import numpy as np
import torch
import warp as wp
from viserdex.supermimic.simulation import Adapter

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

if '--evaluate' not in sys.argv or '--resume' not in sys.argv:
    raise ValueError('Read-only diagnostic requires --evaluate and --resume')
probe_seed=int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else 42
original=Adapter.step
count=0
out=Path(os.environ['RESPONSE_OUTPUT']);out.mkdir(parents=True,exist_ok=True)
def step(self,action):
    global count
    probe_steps=[int(v) for v in os.environ.get('RESPONSE_STEPS',os.environ.get('RESPONSE_STEP','128')).split(',')]
    if count in probe_steps:
        from isaaclab_newton.physics import NewtonManager as N
        snap=Snapshot([self,N.backend,N._solver,N._contacts,N._collision_pipeline])
        print('SNAPSHOT',len(snap.tensors),len(snap.arrays),len(snap.objects),flush=True)
        half=(self.limits[:,:,1]-self.limits[:,:,0])/2
        idx=torch.arange(len(action),device=action.device)
        direction_offset=int(os.environ.get('RESPONSE_DIRECTION_OFFSET','0'))
        if len(probe_steps)>1:direction_offset+=5*probe_steps.index(count)
        j=(idx+direction_offset)%21
        branch_out=out/str(count) if len(probe_steps)>1 else out
        branch_out.mkdir(parents=True,exist_ok=True)
        eps=float(os.environ.get('RESPONSE_EPS_DEG','.5'))
        delta=eps*torch.pi/180/half[idx,j];eligible=action[idx,j].abs()+delta<1
        for label,sign in [('base',0),('repeat',0),('repeat2',0),('repeat3',0),('plus',1),('minus',-1),('minus2',-1),('plus2',1)]:
            snap.restore(); rows=[]
            for offset in range(8):
                command=action.clone()
                if offset==0:command[idx,j]+=sign*delta*eligible
                result=original(self,command)
                rows.append({k:v.detach().cpu().clone() for k,v in result[0].items()})
            torch.save(dict(records=rows,context=self.context.cpu(),limits=self.limits.cpu(),sign=sign,eps_deg=eps,probe_step=0,source_step=count,seed=probe_seed,direction_offset=direction_offset),branch_out/(label+'.pt'))
        snap.restore()
    count+=1
    return original(self,action)
Adapter.step=step
runpy.run_path(str(Path(__file__).with_name('train.py')),run_name='__main__')
