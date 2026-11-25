import torch
from viserdex.supermimic.replay import Replay
from viserdex.supermimic.core import STATE_DIM

def transition(t, done=False):
    s=torch.zeros(2,STATE_DIM); s[:,48]=1; s[:,166]=1
    n=s[:,:55].clone(); n[:,42]=.25
    return dict(state=s, next_physical=n, action=torch.full((2,21),float(t)), done=torch.full((2,),done))

def test_archive_excludes_reset_and_keeps_terminal():
    r=Replay(16,2,'cpu',event_capacity=12,event_fraction=1)
    for i in range(8): r.add(transition(i, i==7))
    assert r.archive_size==2
    assert r.sample(10,8)['done'][:,-1].all()
    assert not r.sample(10,8)['done'][:,:-1].any()
    for i in range(7): r.add(transition(8+i))
    assert r.archive_size==2
    r.add(transition(15)); assert r.archive_size==4
    assert r.archive['action'][2,0,0]==8

def test_empty_fallback_and_bounded_storage():
    r=Replay(16,2,'cpu',event_capacity=3)
    for i in range(7): r.add(transition(i))
    assert r.archive_size==0
    r.add(transition(7)); assert r.sample(8,8)['mask'].shape==(8,8)
    for i in range(8,40): r.add(transition(i))
    assert r.archive_size==3
    assert set(r.archive)=={'state','next_physical','action','done','env_id'}
    assert r.sample(16)['state'].shape==(16,STATE_DIM)


def test_ramp_boundaries():
    from viserdex.supermimic.trainer import actor_ramp
    assert actor_ramp(499,500,1000)==0
    assert actor_ramp(500,500,1000)==.001
    assert actor_ramp(1499,500,1000)==1
    assert actor_ramp(1500,500,1000)==1
    assert actor_ramp(500,500,0)==1


def test_holdout_includes_terminal_then_masks_reset():
    from viserdex.supermimic.validation import open_loop_errors
    class Persistence:
        def __call__(self, s, target, context):
            return s[:, :55]
    rows=[transition(t, t==0) for t in range(3)]
    for row in rows:
        row['next_physical']=row['state'][:,:55].clone()
    result=open_loop_errors(Persistence(), rows, torch.zeros(2,1), torch.tensor([-1.,1.]).expand(2,21,2))
    assert result['1']['valid_environments']==2
    assert result['1']['reward_mae']==0
    assert result['2']['valid_environments']==0
    assert result['2']['reward_mae'] is None
