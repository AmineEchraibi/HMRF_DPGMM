import numpy as np
from numpy import linalg as LA
from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.special import digamma
#import torch as t
import matplotlib.pyplot as plt
#from torch.utils.data import Dataset
#import torch
import math
from matplotlib.patches import Ellipse
import matplotlib.transforms as t
ex = np.expand_dims

eps = 1e-9

def is_invertible(a):
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def cumsum_ex(arr):
    """
    Function computing the cumulative sum exluding the last element for the first element the cumsum is 0
    :param arr: array of shape [p,]
    :return: cum_sum_arr: of shape [p,] where cum_sum_arr[0] = 0 and cum_sum_arr[i] = cumsum(arr[:i])
    """
    cum_sum_arr = np.zeros_like(arr)
    for i in range(len(arr)):
        if i == 0:
            cum_sum_arr[i] = 0
        else:
            cum_sum_arr[i] = np.cumsum(arr[:i])[-1]
    return cum_sum_arr


def get_tuples(dict_N):
    tuples = []
    for n in dict_N.keys():
        for m in dict_N[n]:
            if (n, m) not in tuples and (m, n) not in tuples:
                tuples.append((n, m))
    return tuples


def construct_neighborhood_tree(partial_labels):
    """
    Constructing the neigborhood structure in the form of trees
    :param partial_labels:
    :return: dict_N : dictionary for each index its neighbors list or empty if there is no neighbor
    :return mask: The mask indicating if the neighborhood is empty. [N,]
    """
    mask = np.ones((len(partial_labels),))
    dict_N = {}
    None_indexes = [i for i, v in enumerate(partial_labels) if v == None]
    for i in None_indexes:
        dict_N[i] = []
        mask[i] = 0

    uniques = np.unique(remove_values_from_list(partial_labels, None))
    for u in uniques:
        u_indexes = np.where(partial_labels == u)[0]
        random_root = np.random.choice(u_indexes)
        u_indexes_minus_root = np.delete(u_indexes, np.where(u_indexes == random_root)[0])
        dict_N[random_root] = u_indexes_minus_root
        for leaf in u_indexes_minus_root:
            dict_N[leaf] = np.array([random_root])

    return dict_N, mask


def construct_neighborhood_complete_graph(partial_labels):
    """
    Function constructing the neighborhoods of the MRF in the form of complete graphs
    :param partial_labels:
    :return: dict_N : dictionary for each index its neighbors list or empty if there is no neighbor
    :return mask: The mask indicating if the neighborhood is empty. [N,]
    """
    mask = np.ones((len(partial_labels),))
    dict_N = {}
    dict_N_bar = {}
    None_indexes = [i for i, v in enumerate(partial_labels) if v == None]
    for i in None_indexes:
        mask[i] = 0

    for i in range(len(partial_labels)):
        label = partial_labels[i]
        dict_N[i] = []
        dict_N_bar[i] = []
        if label != None:
            for j in range(len(partial_labels)):
                if j!= i and label == partial_labels[j]:
                    dict_N[i].append(j)


    return dict_N, mask


def remove_values_from_list(the_list, val):
    return [value for value in the_list if value != val]


def cluster_acc(Y_pred, Y):
    """
    Function computing cluster accuracy and confusion matrix at a permutation of the labels
    :param Y_pred: The predicted labels of shape [N, ]
    :param Y: The true labels of shape [N, ]
    :return: Clusterring_accuracy, Confusion matrix
    """
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
        ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size, w


def get_ind_function(X):
    """
    Returns a dictionnary of indicator function for values taken by the random variables over the vocabulary of each dimension
    :param X: data frame of shape [N, d]
    :return: dict_C: for each X_i the indicator of values taken of shape [N, |X_i|]
    """
    dict_C = {}
    N = X.shape[0]
    for column in X.columns:
        vocabulary = np.unique(X[column])
        C = np.zeros(shape=(N, len(vocabulary)))
        for i in range(len(vocabulary)):
            v = vocabulary[i]
            C[:, i] = (X[column].values == v).astype(np.int32)
        dict_C[column] = C

    return dict_C

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def step_decay(epoch):
   initial_lrate = 0.01
   drop = 1
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate


def normalize(params, axis=0):
    """
    Function normalizing the parameters vector params with respect to the Axis: axis
    :param params: array of parameters of shape [axis0, axis1, ..., axisp] p can be variable
    :return: params: array of same shape normalized
    """

    return params / np.sum(params, axis=axis, keepdims=True)


def diagonalize_matrix(A):
    """
    Function returning the diagonalisation of matrix A
    :param A: matrix to be diagonalized
    :return: P: the transition matrix
    :return: D: diagonal matrix
    """
    w, P = LA.eig(A)
    return P, np.diag(w)

def get_batch(X, phi, M):

    N = X.shape[0]
    valid_indices = np.array(range(N))
    batch_indices = np.random.choice(valid_indices,size=M,replace=False)
    batch_x = X[batch_indices,:]
    batch_phi = phi[batch_indices, :]
    return batch_x, batch_phi

def stochastic_update(params_h, old_params, pho):

    new_params = {}
    for key in params_h.keys():
        new_params[key] = (1- pho) * old_params[key] + pho * params_h[key]

    return new_params

def transition_matrix_to_base(A, P):
    """
    Transition matrix to a certain base using the transition matrix P
    :param A: matrix to be transitioned
    :param P: transition matrix
    :return: A_t: transitioned matrix
    """

    return np.dot(LA.inv(P), np.dot(A, P))


def gradient_ascent(init_p, compute_grad, max_iters=100, learning_step=0.01, precision=1e-15, gamma=0.9):
    """
    Function preforming gradient descent
    :param init_p: initial value of the parameters array_like
    :param precision: Desired precision of result
    :param grad_compute: function to compute the gradient at each iteration
    :param max_iters: maximal number of iterations
    :param learning_step: learning step size
    :return: p: array_like of the same shape as init_p
    """
    next_p = init_p
    velocity = np.zeros_like(next_p)
    for i in range(max_iters):
        current_p = next_p
        grad = compute_grad(current_p)
        velocity = gamma * velocity + learning_step * grad
        next_p = current_p + velocity
        diff = next_p - current_p
        print(LA.norm(diff.flatten(), ord=np.inf))
        if LA.norm(diff.flatten(), ord=np.inf) < precision:
            break

    return next_p

def plot(X, y, mus):
    fig = plt.figure(1)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='viridis')
    for k in range(mus.shape[0]):
        plt.plot(mus[k, 0], mus[k, 1],'kx')

    return fig

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = t.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def multivar_digamma(nu, d):
    if np.array(nu).shape == ():
        nu = np.reshape(nu, [1])
    K = np.array(nu).shape[0]
    return np.sum(digamma(0.5 * (nu.reshape(1, K) + 1 - np.arange(1, d + 1).reshape(d, 1))), 0).squeeze()