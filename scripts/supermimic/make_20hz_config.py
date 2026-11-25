"""Derive a bound 20 Hz environment from the saved validated 30 Hz environment."""
import argparse
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'source/viserdex'))
from viserdex.distillation.environment import load_environment
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('source', type=Path)
p.add_argument('output', type=Path)
a=p.parse_args()
d=load_environment(a.source)
assert abs(d['sim']['dt']-1/120)<1e-10 and d['decimation']==4
assert d['episode_length_s']==20
assert abs(d['rewards']['success_bonus']['weight']-250)<1e-8
assert abs(d['rewards']['action_rate_l2']['weight']+.01)<1e-8
assert abs(d['actions']['joint_pos']['alpha']-.95)<1e-8
# Keep the physics step and physical assets; convert only control-time semantics.
d['decimation']=6
d['sim']['render_interval']=6
d['actions']['joint_pos']['alpha']=1-(1-.95)**1.5
d['rewards']['success_bonus']['weight']=250*20/30
d['rewards']['action_rate_l2']['weight']=-.01*(20/30)**2
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:
    yaml.dump(d,f,sort_keys=False)
print(a.output)
