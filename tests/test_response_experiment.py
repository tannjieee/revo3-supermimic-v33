"""Guard experimental response supervision against reset/noise contamination."""
import importlib.util
from pathlib import Path
import pytest
import torch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/supermimic/experiment_v30_response_fit.py'
spec=importlib.util.spec_from_file_location('response_experiment',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def write_branches(folder, *, noisy=False, mismatch=False):
    for name,sign in [('base',0),('repeat',0),('repeat2',0),('repeat3',0),('plus',1),('minus',-1),('plus2',1),('minus2',-1)]:
        rows=[]
        for step in range(8):
            state=torch.zeros(2,84);state[:,48]=1;state[:,82]=1
            if mismatch and name=='plus':state[0,0]=.1
            nxt=state[:,:55].clone();angle=torch.full((2,),sign*.01)
            if noisy and name=='repeat3':angle[:]=.02
            nxt[:,47]=torch.sin(angle/2);nxt[:,48]=torch.cos(angle/2)
            action=torch.zeros(2,21);action[:,0]=sign*.1
            done=torch.zeros(2,dtype=torch.bool)
            # After this boundary the second world's data belong to a reset episode.
            if step==3:done[1]=True
            rows.append(dict(state=state,next_physical=nxt,action=action,done=done,terminated=done.clone(),reward=torch.zeros(2)))
        torch.save(dict(records=rows,context=torch.ones(2,65),limits=torch.tensor([-1.,1.]).expand(2,21,2)),folder/(name+'.pt'))


def test_response_excludes_boundary_and_reset_tail(tmp_path):
    write_branches(tmp_path);data=module.load(tmp_path,'cpu')
    assert data['resolved'][:,0].all()
    assert data['resolved'][:3,1].all()
    assert not data['resolved'][3:,1].any()
    # Ordinary WM supervision still includes the terminal transition itself.
    w=module.window(data,torch.arange(2))
    assert w['mask'][1,3]==1 and w['mask'][1,4]==0


def test_repeat_noise_is_not_response_supervision(tmp_path):
    write_branches(tmp_path,noisy=True)
    assert not module.load(tmp_path,'cpu')['resolved'].any()


def test_different_initial_state_is_rejected(tmp_path):
    write_branches(tmp_path,mismatch=True)
    with pytest.raises(AssertionError):module.load(tmp_path,'cpu')


def test_training_and_test_snapshot_overlap_is_rejected(tmp_path,monkeypatch):
    write_branches(tmp_path)
    output=tmp_path/'output'
    monkeypatch.setattr('sys.argv',['experiment','--checkpoint',str(tmp_path/'unused.pt'),
        '--train',str(tmp_path),'--test',str(tmp_path),'--output',str(output)])
    with pytest.raises(ValueError,match='disjoint'):module.main()
    assert not output.exists()
