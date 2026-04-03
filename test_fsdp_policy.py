from credit.models import load_fsdp_or_checkpoint_policy

conf = {"model": {"type": "wxformer"}}

cls = load_fsdp_or_checkpoint_policy(conf)
print("FSDP policy classes:", sorted([c.__name__ for c in cls]))
print("OK")
