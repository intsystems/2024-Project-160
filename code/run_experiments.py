import numpy as np
from importlib import reload
from sklearn.datasets import load_svmlight_file

import utils
reload(utils)

def init_experiment(func_name, d=112, seed=18, sigma=0, c=False, **init_args):
    args = {}
    if func_name == "quadratic":
        np.random.seed(seed)
        L = init_args['L']
        mu = init_args['mu']
        args['A'] = utils.generate_matrix(d, mu, L)
        args['b'] = np.random.random(d)
        if c:
            args['c'] = np.random.random(1)
        else:
            args['c'] = np.zeros(1)
            
    elif func_name == "mushrooms":
        dataset = "mushrooms.txt" 
        data = load_svmlight_file(dataset)
        X, y = data[0].toarray(), data[1]
        y = y * 2 - 3
        matrix = X * np.expand_dims(y, axis=1)
        args['matrix'] = matrix
    return args
