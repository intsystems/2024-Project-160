import numpy as np
import tqdm
from importlib import reload

import gradient_approximation
import sets
import utils
reload(gradient_approximation)
reload(sets)
reload(utils)

class GDOptimizer:
    def __init__(self, gradient_approximator, learning_rate_k, x_0, sett=None,
                 x_sol=None, max_oracle_calls=10**4, tol=0.000001, seed=18):
        np.random.seed(seed)
        self.gradient_approximator = gradient_approximator # instnce *Aproximator
        self.set = sett # instnce L2Ball, L1Ball, Simplex or None if no projection
        self.learning_rate_k = learning_rate_k
        self.x_0 = x_0
        self.x_curr = np.copy(self.x_0)
        self.d = len(x_0)
        self.x_sol = x_sol
        try:
            self.func_name = gradient_approximator.ZO_oracle.func_name
            if self.func_name == "quadratic":
                self.A = gradient_approximator.ZO_oracle.A
                self.b = gradient_approximator.ZO_oracle.b
                self.c = gradient_approximator.ZO_oracle.c
            elif self.func_name == "mushrooms":
                self.matrix = gradient_approximator.ZO_oracle.matrix
                self.alpha = gradient_approximator.ZO_oracle.alpha
        except AttributeError:
            self.func_name = gradient_approximator.func_name
            if self.func_name == "quadratic":
                self.A = gradient_approximator.A
                self.b = gradient_approximator.b
                self.c = gradient_approximator.c
            if self.func_name == "mushrooms":
                self.matrix = gradient_approximator.matrix    
                self.alpha = gradient_approximator.alpha

        if x_sol is not None:
            if self.func_name == "quadratic": 
                self.f_sol = utils.quadratic_func(self.x_sol, self.A, self.b, self.c)
            elif self.func_name == "mushrooms":
                self.f_sol = utils.logreg_func(self.x_sol, self.matrix, self.alpha)

        self.R0 = self.get_error(x_0)
        self.max_oracle_calls = max_oracle_calls
        self.tol = tol
        self.name = "GD"

    def step(self, x, k):
        gamma_k = self.learning_rate_k(k)
        nabla_f, oracle_calls = self.gradient_approximator.approx_gradient(x, k)
        x_next = x - gamma_k * nabla_f
        x_next = self.set.projection(x_next)

        return x_next, oracle_calls
    
    def get_error(self, x):
        if self.x_sol is None: #||grad(x_k)||
            if self.func_name == "quadratic":
                error = np.linalg.norm(utils.quadratic_grad(x, self.A, self.b))
            elif self.func_name == "mushrooms":
                error = np.linalg.norm(utils.logreg_grad(x, self.matrix, alpha=self.alpha))
        else: #||x_k - x_sol||
            if self.func_name == "quadratic":
                error = np.linalg.norm(x - self.x_sol)
            if self.func_name == "mushrooms":
                error = np.linalg.norm(x - self.x_sol)
        return error
    
    def optimize(self):
        if self.gradient_approximator.name == "JAGUAR":
            batch_size = self.gradient_approximator.batch_size
            num_iter = (self.max_oracle_calls - 2 * self.d) // batch_size + 1
        if self.gradient_approximator.name == "$l_2$-smoothing":
            num_iter = self.max_oracle_calls // 2
        if self.gradient_approximator.name == "full-approximation":
            num_iter = self.max_oracle_calls // (2 * self.d)
        if self.gradient_approximator.name == "True grad":
            num_iter = self.max_oracle_calls // (2 * self.d)
        if self.gradient_approximator.name == "coordinate-approximation":
            num_iter = self.max_oracle_calls // 2
        
        self.oracle_calls_list = [0]
        self.errors_list = [1.]
        for k in tqdm.trange(num_iter):
            self.x_curr, oracle_calls = self.step(self.x_curr, k)
            self.oracle_calls_list.append(self.oracle_calls_list[-1] + oracle_calls)
            error = self.get_error(self.x_curr) / self.R0
            self.errors_list.append(error)
            if error <= self.tol:
                print(f"Precision {self.tol} achieved at step {k}!")
                break

class FWOptimizer(GDOptimizer):
    def __init__(self, gradient_approximator, learning_rate_k, 
                 x_0, sett=None, x_sol=None, max_oracle_calls=10 ** 4, 
                 tol=0.000001, seed=18):
        super().__init__(gradient_approximator, learning_rate_k, 
                         x_0, sett, x_sol, max_oracle_calls, tol, seed)
        self.name = "FW"
    
    def step(self, x, k):
        gamma_k = self.learning_rate_k(k)
        nabla_f, oracle_calls = self.gradient_approximator.approx_gradient(x, k)
        s = self.set.fw_argmin(nabla_f)
        x_next = (1 - gamma_k) * x + gamma_k * s

        return x_next, oracle_calls


class AcceleratedGDOptimizer(GDOptimizer):
    def __init__(self, gradient_approximator, learning_rate_k,
                 x_0, theta, eta, beta, p,
                 sett=None, x_sol=None, max_oracle_calls=10 ** 4, 
                 tol=0.000001, seed=18):
        super().__init__(gradient_approximator, learning_rate_k, 
                         x_0, sett, x_sol, max_oracle_calls, tol, seed)
        self.name = "AcceleratedGD"
        self.x_f_curr = np.copy(self.x_0)
        self.theta = theta
        self.eta = eta
        self.beta = beta
        self.p = p
    
    def step(self, x, k):
        x_h_curr = self.theta * self.x_f_curr + (1 - self.theta) * x
        nabla_f, oracle_calls = self.gradient_approximator.approx_gradient(x_h_curr, k)
        x_f_next = x_h_curr - self.p * self.learning_rate_k(k) * nabla_f
        x_next = self.eta * x_f_next + (self.p - self.eta) * self.x_f_curr + (1 - self.p) * (1 - self.beta) * x + (1 - self.p) * self.beta * x_h_curr
        self.x_f_curr = x_f_next

        return x_next, oracle_calls

class NesterovOptimizer(GDOptimizer):
    def __init__(self, gradient_approximator, learning_rate_k, 
                 x_0, momentum, sett=None, x_sol=None, max_oracle_calls=10 ** 4, 
                 tol=0.000001, seed=18):
        super().__init__(gradient_approximator, learning_rate_k, 
                         x_0, sett, x_sol, max_oracle_calls, tol, seed)
        self.name = "Nesterov"
        self.momentum = momentum
        self.delta_x = np.zeros(x_0.shape)
    
    def step(self, x, k):
        gamma_k = self.learning_rate_k(k)
        nabla_f, oracle_calls = self.gradient_approximator.approx_gradient(x + self.momentum * self.delta_x, k)
        x_next = x - gamma_k * nabla_f + self.momentum * self.delta_x
        self.delta_x = x_next - x
        x_next = self.set.projection(x_next)
        return x_next, oracle_calls