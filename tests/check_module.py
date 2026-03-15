import credit.domain_parallel.halo_exchange as m
print(f"Module: {m.__file__}")
import inspect
src = inspect.getsource(m._HaloExchangeFunction.backward)
print(f"Has file write: {'open' in src}")
print(f"First 100 chars: {src[:100]}")
