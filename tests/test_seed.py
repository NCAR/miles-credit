from credit.seed import seed_everything
import numpy as np


def test_seed():
    r_seed = 4321
    seed_everything(seed=r_seed)
    assert r_seed == np.random.get_state()[1][0], "Random seed not set in numpy."
