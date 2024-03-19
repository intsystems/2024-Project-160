import numpy as np
from matplotlib import pylab as plt

def generate_matrix(d, mu, L):
    diag = (L - mu) * np.random.random_sample(d) + mu
    sigma = np.diag(diag)
    sigma[0][0] = L
    sigma[d - 1][d - 1] = mu
    rand_matrix = np.random.rand(d, d)
    rand_ort, _, _ = np.linalg.svd(rand_matrix)
    matrix = rand_ort.T @ sigma @ rand_ort
    return matrix

def quadratic_func(x, A, b, c):
    return 1./2 * x.T @ A @ x - b.T @ x + c

def quadratic_grad(x, A, b):
    return A @ x - b

def logreg_func(x, matrix, alpha = 0.1):
    x_exp = np.expand_dims(x, axis=1)
    loss = np.mean(np.log(1 + np.exp(- matrix @ x_exp))) + alpha/2 * np.square(np.linalg.norm(x))
    return loss

def logreg_grad(x, matrix, alpha = 0.1):
    x_exp = np.expand_dims(x, axis=1)
    grad = - np.mean(matrix / (1 + np.exp(matrix @ x_exp)), axis = 0)
    return grad + alpha * x

def make_err_plot(optimizers_list, labels=None, title=None, markers=None, colors=None, save_name=None):
    if markers is None:
        markers = [None] * 100
    if colors is None:
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'black', 'olive', 'pink', 'brown']

    x_label = "The number of oracle calls"
    if optimizers_list[0].x_sol is not None:
        y_label = r'$\frac{||x^k - x^*||}{||x^0 - x^*||}$'
    else:
        y_label = r'$||\nabla f(x^k)||$'

    if labels is None:
        if len(set([optimizer.gradient_approximator.name for optimizer in optimizers_list])) > 1:
            labels = [optimizer.gradient_approximator.name for optimizer in optimizers_list]
            if title is None:
                pass
                #mode = optimizers_list[0].gradient_approximator.oracle_mode
                # title = f"Various approximations of the gradient, {optimizers_list[0].name} algorithm"
        elif len(set([optimizer.name for optimizer in optimizers_list])) > 1:
            labels = [optimizer.name for optimizer in optimizers_list]
            # if title is None:
            #    title = f"Various algorithms, {optimizers_list[0].gradient_approximator.name} approximation"
        else:
            raise ValueError("Enter labels to the plot!")

    if title is not None:
        plt.title(title + "\n logarithmic scale on the axis y")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for optimizer, label, color, marker in zip(optimizers_list, labels, colors, markers):
        oracles_calls = optimizer.oracle_calls_list
        errors = optimizer.errors_list
        plt.semilogy(oracles_calls, errors, color=color, label=label, markevery=100, marker=marker)

    plt.legend()
    plt.grid(True)
    
    if save_name is not None:
        plt.savefig(f"figures/{save_name}.pdf", format='pdf')
        plt.savefig(f"figures/{save_name}.png", format='png')
        
    plt.show()
    
def plot_graphs(data, columns_to_plot, save_name=None, plot_params=None, ticks_params=None):

    for column in columns_to_plot:
        params = plot_params.get(column, {})
        plt.plot(data['Step'], data[column], label=params.get('label', column), markevery=100,
                 marker=params.get('marker', 'o'),
                 color=params.get('color', None))

    plt.xlabel('Information sent')
    plt.ylabel(r'$\frac{f(x^k)-f(x^*)}{f(x^0) - f(x^*)}$')
    plt.title('Convergence of GD of MNIST')
    plt.legend()
    plt.grid(True)

    if save_name is not None:
        plt.savefig(output_filename + '.pdf', format='pdf')
        plt.savefig(output_filename + '.png', format='png')

    plt.show()

