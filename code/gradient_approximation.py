import numpy as np
from importlib import reload

import utils
reload(utils)

class ZO_oracle:
    def __init__(self, func_name="quadratic", sigma=0, oracle_mode="tpf", args=None):
        self.func_name=func_name
        if func_name not in ["quadratic", "mushrooms"]:
            raise ValueError(f"Wrong function name {func_name}!")
        self.sigma = sigma
        self.oracle_mode = oracle_mode
        if oracle_mode not in ["opf", "tpf", "det"]:
            raise ValueError(f"Wrong oracle mode {oracle_mode}!")
        self.args = args
        if self.func_name == "quadratic":
            self.A = self.args['A']
            self.b = self.args['b']
            self.c = self.args['c']
        elif self.func_name == "mushrooms":
            self.matrix = self.args['matrix']
            self.alpha = self.args['alpha']
        self.name = f"{oracle_mode} oracle"

    def get_points(self, point_1, point_2):
        if self.func_name == "quadratic":
            if self.oracle_mode in ["opf", "tpf"]:
                A_noise_1 = np.random.normal(loc=0, scale=self.sigma, size=self.A.shape)
                b_noise_1 = np.random.normal(loc=0, scale=self.sigma, size=self.b.shape)
                c_noise_1 = np.random.normal(loc=0, scale=self.sigma, size=self.c.shape)

                if self.oracle_mode == "opf":
                    A_noise_2 = np.random.normal(loc=0, scale=self.sigma, size=self.A.shape)
                    b_noise_2 = np.random.normal(loc=0, scale=self.sigma, size=self.b.shape)
                    c_noise_2 = np.random.normal(loc=0, scale=self.sigma, size=self.c.shape)
                    
                if self.oracle_mode == "tpf":
                    A_noise_2 = np.copy(A_noise_1)
                    b_noise_2 = np.copy(b_noise_1)
                    c_noise_2 = np.copy(c_noise_1)

                func_1 = utils.quadratic_func(point_1, A=self.A + A_noise_1, b=self.b + b_noise_1, c=self.c + c_noise_1)
                func_2 = utils.quadratic_func(point_2, A=self.A + A_noise_2, b=self.b + b_noise_2, c=self.c + c_noise_2)
            if self.oracle_mode == 'det':
                func_1 = utils.quadratic_func(point_1, A=self.A, b=self.b, c=self.c)
                func_2 = utils.quadratic_func(point_2, A=self.A, b=self.b, c=self.c)
                if self.sigma != 0:
                    func_1 = np.round(func_1, int(-np.log10(self.sigma)))
                    func_2 = np.round(func_2, int(-np.log10(self.sigma)))
            
        elif self.func_name == "mushrooms":
            if self.oracle_mode in ["opf", "tpf"]:
                matrix_noise_1 = np.random.normal(loc=0, scale=self.sigma, size=self.matrix.shape)
                
                if self.oracle_mode == "opf":
                    matrix_noise_2 = np.random.normal(loc=0, scale=self.sigma, size=self.matrix.shape)
                    
                if self.oracle_mode == "tpf":
                    matrix_noise_2 = np.copy(matrix_noise_1)
                    
                func_1 = utils.logreg_func(point_1, matrix=self.matrix + matrix_noise_1, alpha=self.alpha)
                func_2 = utils.logreg_func(point_2, matrix=self.matrix + matrix_noise_2, alpha=self.alpha)
            if self.oracle_mode == "det":
                func_1 = utils.logreg_func(point_1, matrix=self.matrix, alpha=self.alpha)
                func_2 = utils.logreg_func(point_2, matrix=self.matrix, alpha=self.alpha)
                if self.sigma != 0:
                    func_1 = np.round(func_1, int(-np.log10(self.sigma)))
                    func_2 = np.round(func_2, int(-np.log10(self.sigma)))

        return func_1, func_2
        
class TrueGradientApproximator:
    def __init__(self, func_name="quadratic", args=None):
        self.func_name = func_name
        self.args = args
        if self.func_name == "quadratic":
            self.A = self.args['A']
            self.b = self.args['b']
            self.c = self.args['c']
        elif self.func_name == "mushrooms":
            self.matrix = self.args['matrix']
            self.alpha = self.args['alpha']

        self.g_curr = None
        self.name = "True grad"
    
    def approx_gradient(self, x, k): ### true gradient ###
        d = len(x)
        if self.func_name == "quadratic":
            grad = utils.quadratic_grad(x, A=self.A, b=self.b)
        elif self.func_name == "mushrooms":
            grad = utils.logreg_grad(x, self.matrix, alpha=self.alpha)
            
        self.g_curr = np.copy(grad)
        
        return self.g_curr, d

class JaguarApproximator:
    def __init__(self, ZO_oracle, gamma=1e-4, momentum_k=None, batch_size=1):
        self.ZO_oracle = ZO_oracle
        self.gamma = gamma
        self.momentum_k = momentum_k
        self.h_curr = None
        self.g_curr = None
        self.batch_size = batch_size
        self.name = "JAGUAR"
        
    def approx_gradient(self, x, k):
        d = len(x)
        if self.h_curr is None:
            self.h_curr = np.zeros_like(x)
            for i in range(d):
                e_i = np.zeros_like(x)
                e_i[i] = 1
                point_1 = x + self.gamma * e_i
                point_2 = x - self.gamma * e_i

                func_1, func_2 = self.ZO_oracle.get_points(point_1, point_2)
                self.h_curr += (func_1 - func_2) / (2 * self.gamma) * e_i
            
            self.g_curr = np.copy(self.h_curr) 
            oracle_calls = d
        else:
            approx_grad = np.zeros_like(x)
            batch_indices = np.random.choice(d, self.batch_size, replace=False)
            for i in batch_indices:
                e_i = np.zeros_like(x)
                e_i[i] = 1
                point_1 = x + self.gamma * e_i
                point_2 = x - self.gamma * e_i
                
                func_1, func_2 = self.ZO_oracle.get_points(point_1, point_2)
                approx_grad = (func_1 - func_2) / (2 * self.gamma) * e_i
                self.h_curr = self.h_curr - self.h_curr[i] * e_i + approx_grad
                
            if self.momentum_k is not None:
                eta_k = self.momentum_k(k)
                self.g_curr = (1 - eta_k) * self.g_curr + eta_k * self.h_curr
            else:
                self.g_curr = np.copy(self.h_curr)
                
            oracle_calls = 2 * self.batch_size

        return self.g_curr, oracle_calls
    
class LameApproximator:
    def __init__(self, ZO_oracle, gamma=1e-5):
        self.ZO_oracle = ZO_oracle
        self.gamma = gamma
        self.g_curr = None
        self.name = "$l_2$-smoothing"
        
    def approx_gradient(self, x, k):
        d = len(x)
        e = np.random.random(size=d)
        e = e / np.linalg.norm(e)
        point_1 = x + self.gamma * e
        point_2 = x - self.gamma * e

        func_1, func_2 = self.ZO_oracle.get_points(point_1, point_2)
        approx_grad = (func_1 - func_2) / (2 * self.gamma) * e
        
        self.g_curr = np.copy(approx_grad)

        return self.g_curr, 1

class TurtleApproximator:
    def __init__(self, ZO_oracle, gamma=1e-5):
        self.ZO_oracle = ZO_oracle
        self.gamma = gamma
        self.g_curr = None
        self.name = "full-approximation"
        
    def approx_gradient(self, x, k):
        d = len(x)
        approx_grad = np.zeros_like(x)
        for i in range(d):
            e_i = np.zeros_like(x)
            e_i[i] = 1
            point_1 = x + self.gamma * e_i
            point_2 = x - self.gamma * e_i

            func_1, func_2 = self.ZO_oracle.get_points(point_1, point_2)
            approx_grad += (func_1 - func_2) / (2 * self.gamma) * e_i
        
        self.g_curr = np.copy(approx_grad)

        return self.g_curr, d

class CoordinateApproximator:
    def __init__(self, ZO_oracle, gamma=1e-5):
        self.ZO_oracle = ZO_oracle
        self.gamma = gamma
        self.g_curr = None
        self.name = "coordinate-approximation"
        
    def approx_gradient(self, x, k):
        d = len(x)
        i = np.random.randint(0, d)
        e_i = np.zeros(d)
        e_i[i] = 1
        point_1 = x + self.gamma * e_i
        point_2 = x - self.gamma * e_i

        func_1, func_2 = self.ZO_oracle.get_points(point_1, point_2)
        approx_grad = (func_1 - func_2) / (2 * self.gamma) * e_i
        
        self.g_curr = np.copy(approx_grad)

        return self.g_curr, 1