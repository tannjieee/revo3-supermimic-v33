"""Fixed-budget WM-only ContactSDF identification with real paired supervision.

Not a policy training checkpoint. Actor/Critic/source weights remain untouched.
Test is opened only after all prespecified fits finish; validation never selects
an update. Per-group robust state and paired response losses use train-only scales.
"""
import argparse,json,sys,time,hashlib,copy,shutil,math
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source/viserdex'))
from experiment_contact_features import dataset,window,Config,physical_delta,action_target,advance,COMPACT,SIZES,GROUPS,score,rollout,sha,save_json
from viserdex.supermimic.contact_sdf import ContactSDF
from viserdex.supermimic.networks import Dynamics


def gradient_failure_details(model):
    """Distinguish nonfinite entries from float32 norm accumulation overflow."""
    result={}
    for name,parameter in model.named_parameters():
        if parameter.grad is None:continue
        gradient=parameter.grad.detach()
        norm=float(gradient.double().norm())
        maximum=float(gradient.abs().max())
        result[name]={'all_entries_finite':bool(torch.isfinite(gradient).all()),
            'norm_float64':norm if math.isfinite(norm) else str(norm),
            'max_abs':maximum if math.isfinite(maximum) else str(maximum)}
    return result


def build(arm,payload,surface,device):
    cfg=Config(**payload['config'])
    if arm=='baseline':
        m=Dynamics(65,cfg);m.load_state_dict(payload['wm'])
    else:m=ContactSDF(payload['geometry_spec'],surface,cfg,payload['metadata'],arm)
    return m.to(device)


def objective(m,d,ids,cfg,state_scale,response_scale):
    w=window(d,ids);n=len(ids);s=w['state'][:,0];ctx=d['context'][ids].repeat(3,1);limits=d['limits'][ids].repeat(3,1,1)
    total=s.new_zeros(());state_total=total;pair_total=total;quiet_total=total
    for t in range(8):
        active=w['mask'][:,t].bool()
        action=torch.where(active[:,None],w['action'][:,t],0.)
        target=action_target(s,action,limits,cfg);pred=m(s,target,ctx)
        truth=torch.where(active[:,None],w['next_physical'][:,t],s[:,:55].detach())
        error=physical_delta(truth,pred)/state_scale
        state_loss=torch.stack([F.smooth_l1_loss(x,torch.zeros_like(x),reduction='none').mean(-1) for x in error.split(SIZES,-1)],-1).mean(-1)
        state_loss=(state_loss*active).sum()/active.sum().clamp_min(1)
        base,plus,minus=pred.split(n)
        difference=physical_delta(minus,plus);real=d['response'][t,ids]
        real_groups=real.split(SIZES,-1);pred_groups=difference.split(SIZES,-1)
        pair=s.new_zeros(());quiet=s.new_zeros(())
        for g,(r,p) in enumerate(zip(real_groups,pred_groups)):
            mask=d['resolved'][t,ids,g]
            delta=(p-r)/response_scale[g]
            value=F.smooth_l1_loss(delta,torch.zeros_like(delta),reduction='none').mean(-1)
            pair+=(value*mask).sum()/mask.sum().clamp_min(1)/6
            quiet_mask=d['valid'][t,ids]&(r.norm(dim=-1)<response_scale[g]*.05)&(d['noise'][t,ids,g]<response_scale[g]*.005)
            value=F.smooth_l1_loss(p/response_scale[g],torch.zeros_like(p),reduction='none').mean(-1)
            quiet+=(value*quiet_mask).sum()/quiet_mask.sum().clamp_min(1)/6
        state_total=state_total+state_loss/8;pair_total=pair_total+pair/8;quiet_total=quiet_total+quiet/8
        nxt=advance(s,pred,action,target,s[:,COMPACT.goal_rot],500,hold_cfg=cfg)
        # Dead branch suffixes cannot contaminate later forward computations.
        s=torch.where(active[:,None],nxt,w['state'][:,0])
    total=state_total+.5*pair_total+.1*quiet_total
    return total,torch.stack((state_total.detach(),pair_total.detach(),quiet_total.detach()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=2000);p.add_argument('--batch_starts',type=int,default=8);p.add_argument('--seeds',type=int,nargs='+',default=[75,76]);p.add_argument('--completed_from',type=Path,help='Reuse only hash-verified completed fits; failed fits replay from initialization for instrumentation');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    a.output.mkdir(parents=True,exist_ok=False);device='cuda:0'
    plan=json.loads((a.experiment/'plan.json').read_text());payload=torch.load(plan['checkpoint'],weights_only=False,map_location='cpu');cfg=Config(**payload['config']);surface=json.loads((a.experiment/'surface.json').read_text())
    train=dataset([a.experiment/f'branches_seed{s}' for s in plan['train_seeds']],device);val=dataset([a.experiment/f'branches_seed{s}' for s in plan['validation_seeds']],device)
    assert set(train['cluster'].tolist()).isdisjoint(val['cluster'].tolist())
    floor=torch.tensor([.003]*21+[.05]*21+[.003]*3+[.005]*3+[.02]*3+[.05]*3,device=device)
    state_scale=torch.maximum((payload['wm']['delta_norm.variance'].to(device)+1e-8).sqrt(),floor)
    response_floor=torch.tensor([.001,.05,.00005,.0005,.005,.01],device=device);response_scale=[]
    for g,r in enumerate(train['response'].split(SIZES,-1)):
        selected=r[train['resolved'][...,g]]
        response_scale.append(torch.maximum(selected.square().mean().sqrt(),response_floor[g]) if len(selected) else response_floor[g])
    response_scale=torch.stack(response_scale)
    report={'status':'fitting','updates':a.updates,'batch_state_world_starts':a.batch_starts,'sampling_seeds':a.seeds,'arms':['quasi','dynamic','baseline'],'source_checkpoint':plan['checkpoint'],'source_checkpoint_sha256':sha(plan['checkpoint']),'script_sha256':sha(__file__),'module_sha256':sha(ROOT/'source/viserdex/viserdex/supermimic/contact_sdf.py'),'surface_sha256':sha(a.experiment/'surface.json'),'test_opened_after_all_prespecified_attempts':True,'physics_changed':False,'actor_critic_updated':False,'state_scale':state_scale.tolist(),'response_scale_training_only':response_scale.tolist(),'loss':'mean group Huber state + 0.5 resolved paired Huber + 0.1 low-noise quiet Huber','learning_rates':{'baseline':1e-5,'quasi':.001,'dynamic':.001},'scope':'Baseline starts from pretrained model40; structured models start from physical parameters. This is extra-budget identification, not an unbiased from-scratch capacity comparison. Seeds change minibatch order only.','fits':{}}
    if a.completed_from:
        prior=json.loads((a.completed_from/'result.json').read_text())
        for field in ('updates','batch_state_world_starts','sampling_seeds','arms','source_checkpoint_sha256','module_sha256','surface_sha256','state_scale','response_scale_training_only','learning_rates'):
            if prior[field]!=report[field]:raise ValueError(f'Cannot reuse different fit contract: {field}')
        report['previous_attempt']={'path':str(a.completed_from),'result_sha256':sha(a.completed_from/'result.json'),
            'scope':'Completed weights reused exactly; incomplete attempts restarted with unchanged optimizer/loss and added failure instrumentation'}
        for key,entry in prior['fits'].items():
            if not entry.get('complete'):continue
            weight=a.completed_from/(key+'_wm_only.pt')
            if sha(weight)!=entry['weights_sha256']:raise ValueError('Completed weight hash mismatch')
            shutil.copy2(weight,a.output/weight.name);report['fits'][key]=copy.deepcopy(entry)
            report['fits'][key]['reused_completed_from']=str(a.completed_from)
    save_json(a.output/'result.json',report)
    for seed in a.seeds:
        batch_generator=torch.Generator().manual_seed(seed);batches=torch.randint(len(train['context']),(a.updates,a.batch_starts),generator=batch_generator)
        for arm in report['arms']:
            key=f'{arm}_seed{seed}'
            if report['fits'].get(key,{}).get('complete'):continue
            m=build(arm,payload,surface,device);opt=torch.optim.Adam(m.parameters(),lr=report['learning_rates'][arm]);entry={'trace':[],'validation':{},'parameter_count':sum(p.numel() for p in m.parameters())};start=time.monotonic()
            entry['validation']['0']=score(rollout(m,val,cfg),val)
            for i,ids in enumerate(batches):
                opt.zero_grad(set_to_none=True);loss,terms=objective(m,train,ids.to(device),cfg,state_scale,response_scale)
                try:
                    if not torch.isfinite(loss):raise FloatingPointError(f'{key} update {i+1}: nonfinite objective')
                    loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
                except (FloatingPointError,RuntimeError) as error:
                    # A failed arm must not cancel independent comparisons. Preserve the
                    # failure; never zero the bad gradient or disable the protection.
                    if isinstance(error,RuntimeError) and 'non-finite' not in str(error):raise
                    entry.update(complete=False,failed=True,failure_update=i+1,failure=str(error),
                        gradient_failure_details=gradient_failure_details(m),elapsed_s=time.monotonic()-start)
                    torch.save({'experimental_only':True,'resumable_policy':False,'failed_update':i+1,
                        'state_dict':m.state_dict(),'optimizer':opt.state_dict(),
                        'gradients':{name:p.grad for name,p in m.named_parameters()}},a.output/(key+'_failed_diagnostic.pt'))
                    report['fits'][key]=entry;save_json(a.output/'result.json',report)
                    print('FIT_FAILED',key,json.dumps(entry['gradient_failure_details']),flush=True)
                    break
                opt.step()
                if (i+1)%50==0 or i==0:
                    row={'update':i+1,'loss':float(loss.detach()),'state_pair_quiet':terms.tolist(),'unclipped_grad_norm':float(grad),'elapsed_s':time.monotonic()-start}
                    entry['trace'].append(row);report['fits'][key]=entry;save_json(a.output/'result.json',report);print(key,json.dumps(row),flush=True)
                if (i+1) in (500,a.updates):entry['validation'][str(i+1)]=score(rollout(m,val,cfg),val)
            if entry.get('failed'):
                del m,opt
                continue
            weight=a.output/(key+'_wm_only.pt');torch.save({'experimental_only':True,'resumable_policy':False,'arm':arm,'config':payload['config'],'state_dict':m.state_dict(),'source_checkpoint':plan['checkpoint'],'surface':surface},weight)
            entry.update(weights_sha256=sha(weight),elapsed_s=time.monotonic()-start,complete=True);report['fits'][key]=entry;save_json(a.output/'result.json',report);print('FIT_COMPLETE',key,flush=True);del m,opt
    report['status']='all_fits_attempted_before_test';save_json(a.output/'result.json',report)
    test=dataset([a.experiment/f'branches_seed{s}' for s in plan['test_seeds']],device)
    assert set(test['cluster'].tolist()).isdisjoint(torch.cat((train['cluster'],val['cluster'])).tolist())
    for key,entry in report['fits'].items():
        if not entry.get('complete'):continue
        saved=torch.load(a.output/(key+'_wm_only.pt'),weights_only=False,map_location='cpu');m=build(saved['arm'],payload,surface,device);m.load_state_dict(saved['state_dict']);prediction=rollout(m,test,cfg);entry['test']=score(prediction,test);torch.save(prediction,a.output/(key+'_test_predictions.pt'));del m
    report['status']='complete_with_failed_fits' if any(v.get('failed') for v in report['fits'].values()) else 'complete';report['source_checkpoint_unchanged']=sha(plan['checkpoint'])==report['source_checkpoint_sha256'];assert report['source_checkpoint_unchanged'];save_json(a.output/'result.json',report)
    print('EXPERIMENT_COMPLETE',flush=True)
if __name__=='__main__':main()
