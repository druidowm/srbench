from occamnet.SKLearn import OccamNet
from multiprocessing import cpu_count
from functools import partial
import numpy as np



def complexity(est: OccamNet, X = None):
    return len(est.best_symbolic)


def model(est: OccamNet, X = None):
    return est.best_symbolic


est = OccamNet(
    num_layers=3,
    equalization=0, 
    num_samples=1000,
    time_limit_hours=0.01, 
    print_every=10,
)

hyper_params = [
    {
        "num_layers": [4],
        "skip_connections": [True],
    }
]
