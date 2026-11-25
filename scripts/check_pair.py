"""Small read/compute NCCL check restricted to the two selected GPUs."""
import os,json,torch
import torch.distributed as dist
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
x=torch.tensor([rank+1.],device='cuda');dist.all_reduce(x)
assert float(x)==3.
y=torch.ones((128,128),device='cuda',dtype=torch.bfloat16);z=y@y
torch.cuda.synchronize();assert torch.isfinite(z).all()
print(json.dumps(dict(rank=rank,device=torch.cuda.get_device_name(rank),nccl_sum=float(x),matmul='passed')),flush=True)
dist.barrier();dist.destroy_process_group()
