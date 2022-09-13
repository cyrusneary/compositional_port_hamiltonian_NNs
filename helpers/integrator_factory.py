import sys
sys.path.append('..')
from integrators.rk4 import rk4

integrators = {
    'rk4': rk4,
}

def integrator_factory(name):
    return integrators[name]