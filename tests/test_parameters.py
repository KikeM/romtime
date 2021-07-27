import numpy as np
from romtime.parameters import get_uniform_dist
from sklearn.model_selection import ParameterSampler


def test_parameter_handler():

    rng = np.random.RandomState(0)

    grid = {
        "alpha": get_uniform_dist(min=1.0, max=10.0),
        "delta": get_uniform_dist(min=-10.0, max=5.0),
        "epsilon": get_uniform_dist(min=0.0, max=1.0),
        "beta": [0.5],
    }

    sampler = ParameterSampler(param_distributions=grid, n_iter=5, random_state=rng)
    param_list = list(sampler)

    rounded_list = [dict((k, round(v, 2)) for (k, v) in d.items()) for d in param_list]

    expected = [
        {"alpha": 5.94, "beta": 0.5, "delta": 0.73, "epsilon": 0.6},
        {"alpha": 5.9, "beta": 0.5, "delta": -3.65, "epsilon": 0.65},
        {"alpha": 4.94, "beta": 0.5, "delta": 3.38, "epsilon": 0.96},
        {"alpha": 4.45, "beta": 0.5, "delta": 1.88, "epsilon": 0.53},
        {"alpha": 6.11, "beta": 0.5, "delta": 3.88, "epsilon": 0.07},
    ]

    assert rounded_list == expected
