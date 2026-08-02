import time
import sys

print("test_subprocess: starting", flush=True)
for i in range(10):
    print(f"test_subprocess: step {i + 1}/10", flush=True)
    time.sleep(1)
print("test_subprocess: done", flush=True)
sys.exit(0)
