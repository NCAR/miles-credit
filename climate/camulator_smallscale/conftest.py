"""Make ``camulator_smallscale`` and ``camulator_gen2`` importable when pytest runs from the repository root.

Neither package is part of the ``miles-credit`` wheel. Installing them (``pip install -e climate/camulator_gen2
-e climate/camulator_smallscale``) is needed to train or roll out, but the tests should run in CI without that step.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _path in (_HERE, os.path.join(os.path.dirname(_HERE), "camulator_gen2")):
    if _path not in sys.path:
        sys.path.insert(0, _path)
