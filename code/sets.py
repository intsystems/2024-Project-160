import numpy as np

class L2Ball:
    def __init__(self, radius):
        self.R = radius
        self.name = "L2 Ball"
    
    def projection(self, x):
        return self.R * x / np.linalg.norm(x)
    
    def fw_argmin(self, nabla_f):
        return - self.R * nabla_f / np.linalg.norm(nabla_f)
        
class R:
    def __init__(self):
        self.name = "R"
    
    def projection(self, x):
        return x
    
class L1Ball:
    def __init__(self, radius):
        self.R = radius
        self.name = "L1 Ball"
    
    def projection(self, x):
        d = len(x)
        y = [0] * d

        def g(llambda, xi):
            summa = 0
            for j in range(d):
                summa += max(np.abs(xi[j]) - llambda, 0)
            return summa - self.R

        if np.linalg.norm(x, ord=1) <= self.R:
            y = x
        else:
            x_sort = np.sort(np.abs(x))
            m = d - 1
            for i, lmbda in enumerate(x_sort):
                if g(lmbda, x) < 0:
                    m = i
                    break
            lmbda = 1. / (d - m) * (np.sum(x_sort[m:]) - self.R)
            for i in range(d):
                y[i] = np.sign(x[i]) * max(np.abs(x[i]) - lmbda, 0)
            y = np.array(y)

        return y
    
    def fw_argmin(self, nabla_f):
        i_max = np.argmax(np.abs(nabla_f))
        ret = np.zeros_like(nabla_f)
        ret[i_max] = -self.R * float(np.sign(nabla_f[i_max]))

        return ret
    
class Simplex:
    def __init__(self):
        self.name = "Simplex"
    
    def projection(self, x):
        x_sort = sorted(x, reverse=True)
        rho = 0
        summa = x_sort[0]
        summa_ans = x_sort[0]

        for i in range(1, len(x_sort)):
            summa += x_sort[i]
            if x_sort[i] + 1 / (i + 1) * (1 - summa) > 0:
                rho = i
                summa_ans = summa

        lamb = 1 / (rho + 1) * (1 - summa_ans)
        x_proj = np.zeros_like(x_sort)

        for i in range(len(x_proj)):
            x_proj[i] = max(x[i] + lamb, 0)

        return x_proj
    
    def fw_argmin(self, nabla_f):
        i_min = np.argmin(nabla_f)
        ret = np.zeros_like(nabla_f)
        ret[i_min] = 1.

        return ret
