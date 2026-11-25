"""Mathematical contracts; these tests do not establish Newton fidelity."""
import torch
from viserdex.supermimic.contact_sdf import smooth_polytope,velocity_projection,rotation_matrix


def test_plane_projection_pushes_both_bodies_and_leaves_separation_unchanged():
    # gap_dot=hand-object; hand velocity -1 pushes the object towards -X.
    normal=torch.tensor([[[1.,-1.]]],dtype=torch.double)/2**.5
    offset=torch.zeros(1,1,dtype=torch.double)
    out=velocity_projection(torch.tensor([[0.,-1.]],dtype=torch.double),normal,offset,1e-8)
    torch.testing.assert_close(out,torch.tensor([[-.5,-.5]],dtype=torch.double))
    free=torch.tensor([[0.,1.]],dtype=torch.double)
    torch.testing.assert_close(velocity_projection(free,normal,offset,1e-8),free)


def test_sdf_signed_inside_and_normal_gradient_agree():
    planes=torch.tensor([[1,0,0,-1],[-1,0,0,-1],[0,1,0,-1],[0,-1,0,-1],[0,0,1,-1],[0,0,-1,-1]],dtype=torch.double)
    q=torch.tensor([[1.2,.1,.2],[.8,.1,.2]],dtype=torch.double,requires_grad=True)
    d,n=smooth_polytope(q,planes,.001)
    torch.testing.assert_close(d,torch.tensor([.2,-.2],dtype=torch.double))
    g,=torch.autograd.grad(d.sum(),q)
    torch.testing.assert_close(n,g)
    assert torch.autograd.gradcheck(lambda x:smooth_polytope(x,planes,.03)[0],(q,))


def test_projection_backward_includes_contact_normals_offsets_and_temperature():
    z=torch.tensor([[.2,-.4]],dtype=torch.double,requires_grad=True)
    n=torch.tensor([[[.8,.6],[-.6,.8]]],dtype=torch.double,requires_grad=True)
    b=torch.tensor([[.1,-.05]],dtype=torch.double,requires_grad=True)
    t=torch.tensor(.04,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(velocity_projection,(z,n,b,t),atol=1e-5)
    assert torch.autograd.gradgradcheck(velocity_projection,(z,n,b,t),atol=1e-5)


def test_xyzw_rotation_matrix_preserves_vector_lengths():
    q=torch.tensor([[0.,0.,2**-.5,2**-.5]],dtype=torch.double)
    r=rotation_matrix(q)
    torch.testing.assert_close(r@torch.tensor([1.,0.,0.],dtype=torch.double),torch.tensor([[0.,1.,0.]],dtype=torch.double))
    torch.testing.assert_close(r@r.transpose(-1,-2),torch.eye(3,dtype=torch.double)[None])
