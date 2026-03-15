"""Check if TF32 causes the conv2 amplification."""
import torch, torch.nn as nn, torch.distributed as dist, os

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

dist.init_process_group(backend='nccl')
rank = int(os.environ.get('LOCAL_RANK', 0))
torch.cuda.set_device(rank)
device = torch.device(f'cuda:{rank}')

from credit.domain_parallel.manager import initialize_domain_parallel
from credit.domain_parallel.convert import convert_to_domain_parallel
from credit.domain_parallel.sharding import shard_tensor, gather_tensor

manager = initialize_domain_parallel(2, 2)
torch.manual_seed(42)

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, 3, padding=1)
        self.norm = nn.GroupNorm(4, 8)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 4, 1)
    def forward(self, x): return self.conv2(self.relu(self.norm(self.conv1(x))))

ref = M().to(device)
dp = M().to(device)
dp.load_state_dict(ref.state_dict())
convert_to_domain_parallel(dp, manager)

if rank == 0:
    inp = torch.randn(1, 4, 16, 32, device=device)
else:
    inp = torch.empty(1, 4, 16, 32, device=device)
dist.broadcast(inp, src=0)

local = shard_tensor(inp, dim=-2, manager=manager)
with torch.no_grad():
    expected = ref(inp)
    dp_local = dp(local)
    dp_out = gather_tensor(dp_local, dim=-2, manager=manager)

if rank == 0:
    diff = (expected - dp_out).abs().max().item()
    print(f'TF32 disabled: max diff = {diff:.2e}')

dist.barrier()
dist.destroy_process_group()
