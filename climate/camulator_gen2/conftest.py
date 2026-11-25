"""Make ``camulator_gen2`` importable when pytest runs from the repository root.

The package is not part of the ``miles-credit`` wheel. Installing it
(``pip install -e climate/camulator_gen2``) is needed to train or roll out, but
the tests should run in CI without that step.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
