import numpy as np
from sklearn.cluster import KMeans
from utils import *
from scipy.special import logsumexp
import matplotlib


from scipy.stats import wishart,dirichlet,beta

def initialize_mixture_parameters(X, partial_labels, K, alpha=1.0):
    """
    Initialise the parameters of the mixture model
    :param X: the data matix of shape [N,K]
    :param partial_labels: the partial labels ranging from {1,..,K'} K' <= K
    :param K: the number of clusters considering.
    :param alpha: the dirichelet distribution concetration prior by default equal to 1/K
    :return: params: dictionary of the parameters of the mixture model
    """
    d = X.shape[1]
    alpha = 1 / K
    kappa_0 = np.ones((K, ))
    epsilon_0 = alpha * np.ones((K, ))
    nu_0 = d  * np.ones((K, ))
    m_0 = np.zeros((K, d))
    L_0 = np.eye(d).reshape(1, d, d) * np.ones((K, 1, 1))
    params = {"kappa_0" : kappa_0, "epsilon_0": epsilon_0, "nu_0":nu_0, "m_0": m_0, "L_0":L_0}
    return params

def initialize_phi(N, K):
    """
    Initializing responsiblities or posterior class probabilities
    :param N: Number of instances
    :param K: Number of classes
    :return: R : responsibilities of shape [N, K]
    """

    phi = np.random.rand(N, K)
    phi = normalize(phi, axis=1)

    return phi

def initialise_phi_with_kmeans(X, K):

    mu = KMeans(K).fit(X).cluster_centers_
    phi = np.exp( - 0.5 * LA.norm(X.reshape(X.shape[0], 1, X.shape[1]) - mu.reshape(1, K, X.shape[1]),2,2))
    return normalize(phi,1), mu

class HMRF_GMM:
    """
    The semi-supervised Dirichelet process Gaussian mixture model
    """

    def __init__(self, X, K, partial_labels, alpha=1.0, epsilon=1e-9, lambda_=1., init="kmeans", weight_prior = "Dirichelet distribution", eta = 1, mode='complete'):
        """
        init function
        :param X: data numpy array of shape [N, d]
        :param K: Trunction level for the approximate betas
        :param partial_labels: array of partial labels where unlabeled data is set to None labeled data is in {1,..., T}
        :param alpha: prior on the Dirichlet dist
        :param epsilon: for stability
        :param lambda_: the hyperparamater for the HMRF
        :param init: initialization Kmeans or random
        :param weight_prior: weight prior Dirichelet distribution or Dirichelet Proceszs
        :param eta: Beta prior
        :param mode: Faster if we consider a tree HMRF on the hidden labels but complete is recommended
        """

        if mode == "tree":
            self.dict_N, self.mask = construct_neighborhood_tree(partial_labels)
        else:
            self.dict_N, self.mask = construct_neighborhood_complete_graph(partial_labels)
        self.N = X.shape[0]
        self.K = K
        self.d = X.shape[1]
        self.params_0 = initialize_mixture_parameters(X,partial_labels,self.K)
        if init == "kmeans":
            self.phi, mu_0 = initialise_phi_with_kmeans(X, K)
            self.params_0["m_0"] = mu_0
        else:
            self.phi = initialize_phi(self.N, K)


        self.eps = epsilon
        self.X = X
        self.lambda_ = lambda_
        self.weight_prior = weight_prior
        self.eta = eta / K
        self.S = 64
        self.to = 1024
        self.delta = 0.5

        self.tuples_ml = get_tuples(self.dict_N)

    def compute_N(self, phi):

        """
        Function computing the sum over all instances of the responsibilities --> N_k = sum_n \phi_{nk}
        :param phi: of shape [N, K]
        :return: N: of shape [1, K]
        """
        return np.sum(phi, 0)

    def compute_nu(self, N):
        """
        The function computing the seconnd variational parameter of the wishart distribution
        :param N: of shape [1, K]
        :return: nu: the second variational parameter of the wishart distributionof shape [K, ]
        """
        nu = self.params_0["nu_0"] + N + 1
        return nu

    def compute_gamma_1(self, N):
        """
        The function computing the first variational parameter of the beta distribution
        :param N: of shape [1, K]
        :return: gamma_1: the first variational parameter of the beta distribution of shape [K, ]
        """
        gamma_1 = 1 + N
        return gamma_1

    def compute_gamma_2(self, N):
        """
        The function computing the second variational parameter of the beta distribution
        :param N: of shape [1, K]
        :return: gamma_2: the second variational parameter of the beta distribution of shape [K, ]
        """
        gamma_2 = self.eta + cumsum_ex(N[::-1])[::-1]
        return gamma_2

    def compute_epsilon(self, N):
        """
        The function computing the first variational parameter of the Dirichelet distribution (if Dirichelet dist prior is considered)
         :param N: of shape [1, K]
        :return: epsilon: the first variational parameter of the Dirichelet distribution of shape [K, ]
        """
        epsilon = self.params_0["epsilon_0"] + N
        return epsilon

    def compute_kappa(self, N):

        """
        The function computing the scalar of the variance for the approximating distribution of the means
        :param N: of shape [1, K]
        :return: kappa: of shape [1, K]
        """

        kappa = self.params_0["kappa_0"] + N
        return kappa

    def compute_m(self, phi, N):

        """
        The function computing the scalar of the mean for the approximating distribution of the means
        :param phi: of shape [N, K]
        :param N: of shape [1, K]
        :return: m: of shape [K, d]
        """


        m = (self.params_0["kappa_0"].reshape(self.K, 1) * self.params_0["m_0"]
             + np.sum(np.reshape(phi, [self.N, self.K, 1]) * np.reshape(self.X, [self.N, 1, self.d]), axis=0) )\
            / (self.params_0["kappa_0"] + N ).reshape(self.K, 1)

        return m

    def compute_L(self, phi, m, nu):

        """
        Function compution the wishart mean matrix of the approximating distribution of the convariance matrices
        :param phi: of shape [N, K]
        :param m: of shape [K, d]
        :param nu: of shape [1, K]
        :return:
        """
        m_0 = np.reshape(self.params_0["m_0"], [self.K, self.d, 1])
        m =  np.reshape(m, [self.K, self.d, 1])

        L_inv = self.params_0["L_0"] + self.params_0["kappa_0"].reshape(self.K, 1, 1) * np.matmul(m - m_0, np.reshape(m - m_0,(self.K, 1, self.d))) \
                + np.sum(np.reshape(phi, [self.N, self.K, 1, 1]) * np.matmul(self.X.reshape(self.N, 1, self.d, 1) - m.reshape(1, self.K, self.d, 1),
                                                                             self.X.reshape(self.N, 1, 1, self.d) - m.reshape(1, self.K, 1, self.d) ) , axis=0)

        return np.linalg.inv(L_inv)

    def compute_phi(self, m, L, epsilon, V, nu, phi_t, kappa, gamma_1, gamma_2):
        """
        Function computing the variational parameters phi
        :param m: of shape [K, d]
        :param L: of shape [K, d, d]
        :param epsilon: of shape [1, K]
        :param V: of shape [K, K]
        :param nu: of shape [1, K]
        :param phi_t: of shape [N, K]
        :param kappa: of shape [1, K]
        :param gamma_1: of shape [1, K]
        :param gamma_2: of shape [1, K]
        :return:
        """

        if self.weight_prior == "Dirichelet distribution":
            val = digamma(epsilon) - digamma(np.sum(epsilon))
        else :
            val = digamma(gamma_1) - digamma(gamma_1 + gamma_2) + cumsum_ex(digamma(gamma_2) - digamma(gamma_1 + gamma_2))

        #print("val", val)

        log_phi = np.zeros((self.N, self.K))

        for n in range(self.N):
            if self.mask[n] == 0:
                log_phi[n, :] = log_phi[n, :] + val
                #print("digmma epsilon ", digamma(epsilon) - digamma(np.sum(epsilon)))
            else:
                for j in self.dict_N[n]:
                    log_phi[n, :] = log_phi[n, :] - self.lambda_ * np.sum(phi_t[j, :].reshape(1, self.K) * V, axis=1)




            log_phi[n, :] = log_phi[n, :] - 0.5 * self.d/kappa - 0.5 * nu * np.trace(np.matmul(L, np.matmul((self.X[n,:].reshape(1, self.d) - m).reshape(self.K, self.d, 1),
                                                                                       (self.X[n,:].reshape(1, self.d) - m).reshape(self.K, 1, self.d) )), axis1=1 , axis2=2) \
                            + 0.5 * (LA.slogdet(L)[1] + multivar_digamma(nu, self.d)) - 0.5 * self.d * np.log(np.pi)



        log_phi = log_phi - logsumexp(log_phi,axis=1)[:,np.newaxis]
        phi = np.exp(log_phi)
        #print(np.sum(log_phi, axis=1).min(0))
        return normalize(phi, axis=1)

    def compute_potentials(self, m, L, nu, kappa):

        """
        Function computing the pairwise potential matrices
        :param m: of shape [K, d]
        :param L: of shape [K, d, d]
        :param nu: of shape [1, K]
        :param kappa: of shape [1, K]
        :return:
        """
        V = np.zeros((self.K, self.K))

        L_inv = LA.inv(L)
        for k in range(self.K - 1):
            for l in range(k + 1, self.K):
                V[k, l] = self.d *(kappa[k]/(self.eps + kappa[l]) - 1) + np.log(kappa[k] / (self.eps + kappa[l])) + nu[k] * \
                            np.trace(np.matmul(L[k,:,:], np.matmul((m[k, :] - m[l, :]).reshape(self.d,1), (m[k, :] - m[l, :]).reshape(1, self.d) ))) \
                            + self.d *(kappa[l]/(self.eps + kappa[k]) - 1) + np.log(kappa[l] / (self.eps + kappa[k])) + nu[l] * \
                            np.trace(np.matmul(L[l,:,:], np.matmul((m[l, :] - m[k, :]).reshape(self.d,1), (m[l, :] - m[k, :]).reshape(1, self.d) ))) \
                            + 0.5 * (nu[k] - nu[l]) * ( np.log(self.eps + LA.det(L[k,:,:])) - np.log(self.eps + LA.det(L[l,:,:]))
                                                       + multivar_digamma(nu[k], self.d) - multivar_digamma(nu[l], self.d)) \
                            + 0.5 * nu[k] * (np.trace(np.matmul(L_inv[l, :,:], L[k,:,:])) - self.d) \
                            + 0.5 * nu[l] * (np.trace(np.matmul(L_inv[k, :,:], L[l,:,:])) - self.d)

        #print("V",  V)
        return V + V.T

    def Inference(self, max_iter=1000, debug=True):
        """
        The gradient ascent algorithm using the fixed point equations
        :param max_iter: Number of max iterations
        :param debug: debug if True
        :return: L: List the evidence lower bound at each iteration
        """
        loss = []
        stop_criterion = False


        #self.mask = np.zeros((self.N,))
            #fig, axs = plt.subplots(1, 10,figsize=(10,3))
            #p=0

        for i in range(max_iter):

            # Fixed point equations for gradient ascent

            N = self.compute_N(self.phi)
            nu = self.compute_nu(N)
            kappa = self.compute_kappa(N)

            m = self.compute_m(self.phi, N)
            L = self.compute_L(self.phi, m, nu)
            V = self.compute_potentials(m, L, nu, kappa)
            #print("phi : ", np.sum(self.phi, axis=0) /self.phi.shape[0])
            if self.weight_prior == "Dirichelet distribution":
                epsilon = self.compute_epsilon(N)
                gamma_1 = None
                gamma_2 = None

            else:
                epsilon = None
                gamma_1 = self.compute_gamma_1(N)
                gamma_2 = self.compute_gamma_2(N)

            phi_t = np.copy(self.phi)
            self.phi = self.compute_phi(m, L, epsilon, V, nu, phi_t, kappa, gamma_1, gamma_2)

            #print(self.phi.shape)
            # Compute evidence lower bound

            l, log_likelihood_term, hmrf_term = self.compute_elbo(self.phi, nu, kappa, epsilon, m, L, V, N, gamma_1, gamma_2)

            if debug:
                print("[DEBUG] elbo at iteration ", i, " is ", l)
                print("[DEBUG] log_likelihood_term at iteration ", i, " is ", log_likelihood_term)
                print("[DEBUG] hmrf_term at iteration ", i, " is ", hmrf_term)
            loss.append(l)

            Z = self.infer_clusters()



            # Stopping criterion
            if len(loss) > 2 :
                stop_criterion = np.abs((loss[-1] - loss[-2]) / loss[-2]) < 1e-9
            if stop_criterion:
                break

        return loss




    def compute_elbo(self, phi, nu, kappa, epsilon, m, L, V, N, gamma_1, gamma_2):
        """
        Function compute the evidence lower bound as defined for HRMF-DPCMM From the variational parameters.
        :param phi: of shape [N,K]
        :param nu: of shape [1, K]
        :param kappa: of shape [1, K]
        :param epsilon: of shape [1, K]
        :param m: of shape [N, d]
        :param L: of shape [N, d, d]
        :param V: of shape [K, K]
        :param N: of shape [1, K]
        :param gamma_1: of shape [1, K]
        :param gamma_2: of shape [1, K]
        :return:
        """
        hmrf_term = 0
        log_likelihood_term = 0

        if self.weight_prior == "Dirichelet distribution":
            val = digamma(epsilon) - digamma(np.sum(epsilon))
        else:
            val = digamma(gamma_1) - digamma(gamma_1 + gamma_2) + cumsum_ex(
                digamma(gamma_2) - digamma(gamma_1 + gamma_2))

        for n in range(self.N):
            for k in range(self.K):
                log_likelihood_term += - 0.5 * phi[n, k] * nu[k] * np.trace(np.matmul(L[k,:,:], np.matmul(self.X[n,:].reshape(self.d, 1) - m[k].reshape(self.d, 1),
                                                                                        self.X[n,:].reshape(1, self.d) - m[k].reshape(1, self.d) )) ) \
                     - 0.5 * self.d * N[k] / kappa[k]
                if self.mask[n] == 0:
                    log_likelihood_term += phi[n,k] * val[k]

        for k in range(self.K):
            log_likelihood_term += 0.5 * (nu[k] - self.d + N[k]) * (multivar_digamma(nu[k], self.d) + np.log(self.eps + LA.det(L[k, :, :]))) - 0.5 * self.d * nu[k]


        for tuple in self.tuples_ml:
            hmrf_term += - self.lambda_ * np.sum(np.sum(phi[tuple[0],:].reshape(self.K, 1) * phi[tuple[1],:].reshape(1,self.K) * V))


        for k in range(self.K):
            log_likelihood_term += wishart(nu[k],L[k,:,:]).entropy() + 0.5*(multivar_digamma(nu[k], self.d) + np.log(self.eps + LA.det(L[k, :, :]))) + 0.5*self.d*np.log(eps + kappa[k]) \
                    - np.sum(phi[:,k] * np.log(phi[:,k] + eps))
            if self.weight_prior != "Dirichelet distribution":
                log_likelihood_term += beta(gamma_1[k], gamma_2[k]).entropy()

        if self.weight_prior == "Dirichelet distribution":
            log_likelihood_term += dirichlet(epsilon).entropy()

        elbo = log_likelihood_term + hmrf_term

        return elbo/self.N, log_likelihood_term, hmrf_term

    def infer_clusters(self):
        """
        Function returning the clustering assignments for each data sample
        :return: y_pred: array of shape [N, ]
        """
        return np.argmax(self.phi, axis=1)
