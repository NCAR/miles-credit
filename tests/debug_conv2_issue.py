"""Debug the conv2 step amplification."""
import torch
import torch.nn as nn
import torch.distributed as dist
import os

def main():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    from credit.domain_parallel.manager import initialize_domain_parallel
    from credit.domain_parallel.convert import convert_to_domain_parallel
    from credit.domain_parallel.sharding import shard_tensor, gather_tensor

    manager = initialize_domain_parallel(2, 2)
    torch.manual_seed(42)

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 8, 3, padding=1)
            self.norm = nn.GroupNorm(4, 8)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(8, 4, 1)
        def forward(self, x):
            x = self.relu(self.norm(self.conv1(x)))
            return self.conv2(x)

    model_ref = SimpleModel().to(device)
    model_dp = SimpleModel().to(device)
    model_dp.load_state_dict(model_ref.state_dict())
    convert_to_domain_parallel(model_dp, manager)

    if rank == 0:
        full_input = torch.randn(1, 4, 16, 32, device=device)
    else:
        full_input = torch.empty(1, 4, 16, 32, device=device)
    dist.broadcast(full_input, src=0)

    local_input = shard_tensor(full_input, dim=-2, manager=manager)

    with torch.no_grad():
        ref_relu_out = model_ref.relu(model_ref.norm(model_ref.conv1(full_input)))
        expected = model_ref.conv2(ref_relu_out)

        dp_relu_out = model_dp.relu(model_dp.norm(model_dp.conv1(local_input)))
        dp_relu_gathered = gather_tensor(dp_relu_out, dim=-2, manager=manager)

        # Test 1: Apply ref conv2 on gathered DP relu output
        if rank == 0:
            test1 = model_ref.conv2(dp_relu_gathered)
            test1_diff = (expected - test1).abs().max().item()
            print(f"Test 1: conv2(gathered_dp_relu) vs ref: {test1_diff:.2e}")

        # Test 2: Apply dp conv2 on gathered DP relu output
        if rank == 0:
            test2 = model_dp.conv2(dp_relu_gathered)
            test2_diff = (expected - test2).abs().max().item()
            print(f"Test 2: dp_conv2(gathered_dp_relu) vs ref: {test2_diff:.2e}")

        # Test 3: DP conv2 on local, then gather
        dp_final_local = model_dp.conv2(dp_relu_out)
        dp_final_gathered = gather_tensor(dp_final_local, dim=-2, manager=manager)
        if rank == 0:
            test3_diff = (expected - dp_final_gathered).abs().max().item()
            print(f"Test 3: gather(dp_conv2(local)) vs ref: {test3_diff:.2e}")

        # Test 4: Check if dp conv2 weights match ref
        if rank == 0:
            w_same = torch.equal(model_ref.conv2.weight, model_dp.conv2.weight)
            b_same = torch.equal(model_ref.conv2.bias, model_dp.conv2.bias)
            print(f"Test 4: weights equal={w_same}, bias equal={b_same}")

        # Test 5: Manual 1x1 conv (matmul) on gathered
        if rank == 0:
            w = model_ref.conv2.weight.squeeze(-1).squeeze(-1)  # (4, 8)
            b = model_ref.conv2.bias  # (4,)
            # ref_relu_out: (1, 8, 16, 32) -> (1, 16*32, 8)
            ref_flat = ref_relu_out.permute(0, 2, 3, 1).reshape(-1, 8)
            dp_flat = dp_relu_gathered.permute(0, 2, 3, 1).reshape(-1, 8)

            ref_manual = ref_flat @ w.T + b
            dp_manual = dp_flat @ w.T + b
            manual_diff = (ref_manual - dp_manual).abs().max().item()
            print(f"Test 5: Manual matmul diff: {manual_diff:.2e}")

            # Check input diff
            input_flat_diff = (ref_flat - dp_flat).abs().max().item()
            print(f"Test 5: Input flat diff: {input_flat_diff:.2e}")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
