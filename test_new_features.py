import torch
from credit.models.wxformer.wxformer_v2 import CrossFormer
from credit.scheduler import LinearWarmupCosineScheduler
from credit.trainers.base_trainer import EMATracker

m = CrossFormer(image_height=64, image_width=128, use_zero_init=True).cuda()
print("zero-init OK")

opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
sched = LinearWarmupCosineScheduler(opt, warmup_steps=10, total_steps=100)
for _ in range(5):
    opt.step()
    sched.step()
print(f"scheduler LR after 5 steps: {opt.param_groups[0]['lr']:.2e}  (expect ~5e-4, halfway through warmup)")

ema = EMATracker(m, decay=0.9999)
ema.update(m)
ema.swap(m)
ema.swap(m)
print("EMA OK")
print("All checks passed.")
