"""Check actual-asset geometry values AND state gradients after compiler fusion."""

import copy
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/viserdex"))
from viserdex.supermimic.geometry import HandObjectGeometry

torch.manual_seed(123)
torch.set_float32_matmul_precision("highest")
spec = json.loads((ROOT / "tests/data/revo3_geometry_v1.json").read_text())
eager = HandObjectGeometry(spec).cuda()
compiled = copy.deepcopy(eager)
compiled.enable_compilation()
state = torch.randn(64, 172, device="cuda") * 0.2
state[:, 45:49] /= state[:, 45:49].norm(dim=-1, keepdim=True)
target = state.detach().clone()
state = state.detach().requires_grad_()
weights = torch.randn(64, eager.feature_dim, device="cuda")
a = eager(state)
b = compiled(state)
torch.testing.assert_close(a, b, atol=5e-5, rtol=5e-5)
ga = torch.autograd.grad((a * weights).sum(), state)[0]
gb = torch.autograd.grad((b * weights).sum(), state)[0]
torch.testing.assert_close(ga, gb, atol=5e-4, rtol=5e-4)
prediction = target[:, :55].clone()
prediction[:, :42] += torch.randn_like(prediction[:, :42]) * 0.03
prediction[:, 42:45] += 0.003
prediction.requires_grad_()
la = eager.supervised_errors(prediction, target)
lb = compiled.supervised_errors(prediction, target)
torch.testing.assert_close(la, lb, atol=1e-4, rtol=1e-4)
gla = torch.autograd.grad(la.mean(), prediction)[0]
glb = torch.autograd.grad(lb.mean(), prediction)[0]
torch.testing.assert_close(gla, glb, atol=5e-4, rtol=5e-4)
result = dict(
    features_max_abs=float((a - b).abs().max()),
    gradient_max_abs=float((ga - gb).abs().max()),
    loss_max_abs=float((la - lb).abs().max()),
    loss_gradient_max_abs=float((gla - glb).abs().max()),
    state_keys_unchanged=list(eager.state_dict()) == list(compiled.state_dict()),
)
assert result["state_keys_unchanged"]
print(json.dumps(result, indent=2))
