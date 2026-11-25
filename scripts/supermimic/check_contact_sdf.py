"""Check complete experimental ContactSDF gradients against numerical differences."""
import argparse,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'source/viserdex'))
from viserdex.supermimic.contact_sdf import ContactSDF
from viserdex.supermimic.core import Config,action_target,physical_delta


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();torch.set_num_threads(4)
    plan=json.loads((a.experiment/'plan.json').read_text());payload=torch.load(plan['checkpoint'],weights_only=False,map_location='cpu')
    surface=json.loads((a.experiment/'surface.json').read_text());cfg=Config(**payload['config'])
    data=torch.load(a.experiment/'branches_seed60/32/base.pt',weights_only=False,map_location='cpu')
    s=data['records'][0]['state'][:2].double();ctx=data['context'][:2].double()
    target=action_target(s,data['records'][0]['action'][:2].double(),data['limits'][:2].double(),cfg)
    report={'status':'running','surface_points':len(surface['point_body']),'models':{}}
    for mode in ('quasi','dynamic'):
        model=ContactSDF(payload['geometry_spec'],surface,cfg,payload['metadata'],mode).double()
        kin=model.kinematics(s);jac=kin[-1];errors=[]
        eps=1e-6
        for j in range(21):
            plus=s.clone();minus=s.clone();plus[:,j]+=eps;minus[:,j]-=eps
            finite=(model.kinematics(plus)[-2]-model.kinematics(minus)[-2])/(2*eps)
            errors.append(float((finite-jac[:,:,:,j]).abs().max()))
        # Fixed joint frames were serialized as float32, including unit quaternions.
        # 0.1 micrometer/radian is below the live FK/mesh readback tolerance.
        assert max(errors)<1e-7,max(errors)
        pred=model(s,target,ctx);assert torch.isfinite(pred).all()
        loss=physical_delta(data['records'][0]['next_physical'][:2].double(),pred).square().mean()
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.zero_grad(set_to_none=True)
        command=target.clone().requires_grad_(True)
        # Weighted physical readout keeps the same loss coordinates on both sides.
        readout=torch.linspace(.3,1.,54,dtype=torch.double)
        scalar=(physical_delta(s,model(s,command,ctx))*readout).sum()
        g,=torch.autograd.grad(scalar,command)
        direction=torch.linspace(-.5,.5,42,dtype=torch.double).reshape(2,21)
        with torch.no_grad():
            plus=(physical_delta(s,model(s,target+eps*direction,ctx))*readout).sum()
            minus=(physical_delta(s,model(s,target-eps*direction,ctx))*readout).sum()
        exact=float((g*direction).sum());finite=float((plus-minus)/(2*eps));error=abs(exact-finite)/max(1.,abs(exact),abs(finite))
        assert error<2e-5,(mode,exact,finite,error)
        phi,jn,jt=model.contact(s,kin)
        report['models'][mode]=dict(fk_jacobian_max_error=max(errors),action_derivative_autograd=exact,
            action_derivative_finite_difference=finite,action_derivative_relative_error=error,
            parameter_gradients_finite=True,first_step_physical_mse=float(loss.detach()),
            minimum_query_gap_m=float(phi.detach().min()),quaternion_norm_max_error=float((pred[:,45:49].detach().norm(dim=-1)-1).abs().max()))
    report['status']='passed';report['scope']='Implementation derivative consistency only; not a certificate of Newton response fidelity or policy readiness.'
    (a.experiment/'full_model_selfcheck.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
