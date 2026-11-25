"""Diagnostic data contract tests; synthetic samples are not physics evidence."""
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/viserdex'))
spec=importlib.util.spec_from_file_location('contact_experiment',ROOT/'scripts/supermimic/experiment_contact_features.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.fixture
def branch_data(tmp_path):
    folder=tmp_path/'32';folder.mkdir()
    (tmp_path/'manifest.json').write_text(json.dumps(dict(status='complete',model_hash_before='unchanged',model_hash_after='unchanged')))
    manifest={'branches':{}}
    for branch in m.BRANCHES:
        sign=1 if branch.startswith('plus') else -1 if branch.startswith('minus') else 0
        records=[]
        for t in range(8):
            s=torch.zeros(1,84);s[:,48]=1;s[:,82]=1
            if t>2:s[:,0]=float('nan') # post-reset data must not enter fit statistics
            action=torch.zeros(1,21);action[:,2]=sign*torch.pi/180 if t<5 else 0
            records.append(dict(state=s,next_physical=s[:,:55].clone(),action=action,done=torch.tensor([t==2]),terminated=torch.tensor([t==2]),reward=torch.zeros(1)))
        obj=dict(records=records,context=torch.zeros(1,65),limits=torch.tensor([[[-1.,1.]]*21]),seed=50,source_step=32,eligible=torch.ones(1,dtype=torch.bool),joint=torch.tensor([2]),amplitudes_deg=torch.tensor([1.]),hold_steps=torch.tensor([5]))
        path=folder/(branch+'.pt');torch.save(obj,path);manifest['branches'][branch]={'sha256':m.sha(path)}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return tmp_path


def rewrite(root,branch,mutate):
    path=root/'32'/(branch+'.pt');d=torch.load(path,weights_only=False);mutate(d);torch.save(d,path)
    p=root/'32'/'manifest.json';v=json.loads(p.read_text());v['branches'][branch]['sha256']=m.sha(path);p.write_text(json.dumps(v))


def test_terminal_transition_in_fit_but_not_postreset(branch_data):
    d=m.dataset([branch_data],'cpu');w=m.window(d,torch.tensor([0]))
    assert w['mask'].tolist()==[[1.,1.,1.,0.,0.,0.,0.,0.]]*3
    assert d['valid'][:,0].tolist()==[True,True,False,False,False,False,False,False]
    class Features:
        feature_norm=m.RunningNorm(1)
        def extra_features(self,s,target,context):return s[:,:1]
    f=Features();cfg=SimpleNamespace(action_mode='direct_limits')
    m.fit_feature_statistics(f,d,cfg)
    assert f.feature_norm.count.item()==9
    assert torch.isfinite(f.feature_norm.mean).all() and torch.isfinite(f.feature_norm.std).all()


def test_unmatched_initial_state_is_rejected(branch_data):
    rewrite(branch_data,'plus',lambda d:d['records'][0]['state'].__setitem__((0,0),.1))
    with pytest.raises(AssertionError):m.dataset([branch_data],'cpu')


def test_wrong_hold_timing_is_rejected(branch_data):
    rewrite(branch_data,'plus',lambda d:d['records'][5]['action'].__setitem__((0,2),.01))
    with pytest.raises(AssertionError):m.dataset([branch_data],'cpu')


def test_corrupt_branch_file_is_rejected(branch_data):
    with (branch_data/'32/base.pt').open('ab') as stream:stream.write(b'corrupt')
    with pytest.raises(AssertionError):m.dataset([branch_data],'cpu')
