import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal


def cov_linear(x, x2=None, theta=1):
    if x2 is None:
        return np.dot(x, x.T) * theta
    else:
        return np.dot(x, x2.T) * theta


def cov_RBF(x, x2=None, theta=np.array([1, 1])):
    """
    Compute the Euclidean distance between each row of X and X2, or between
    each pair of rows of X if X2 is None and feed it to the kernel.
    """
    variance = theta[0]
    lengthscale = theta[1]

    if x2 is None:
        xsq = np.sum(np.square(x), 1)
        r2 = -2. * np.dot(x, x.T) + (xsq[:, None] + xsq[None, :])
        r = np.sqrt(r2) / lengthscale
    else:
        x1sq = np.sum(np.square(x), 1)
        x2sq = np.sum(np.square(x2), 1)
        r2 = -2. * np.dot(x, x2.T) + x1sq[:, None] + x2sq[None, :]
        r = np.sqrt(r2) / lengthscale

    return variance * np.exp(-0.5 * r ** 2)


class CovFunctions(object):
    """
    A wrapper for covariance functions which is compatible with the GP class.
    """

    def __init__(self, covType, theta):
        self.covType = covType
        self.theta = theta
        if covType == 'linear':
            self.compute = self.linear
        elif covType == 'RBF':
            self.compute = self.RBF

    def set_theta(self, theta):
        self.theta = theta

    def get_theta(self):
        return self.theta

    def linear(self, x, x2=None):
        return cov_linear(x, x2, self.theta)

    def RBF(self, x, x2=None):
        return cov_RBF(x, x2, self.theta)


class GP(object):
    def __init__(self, X, Y, sigma2, covType, theta):
        self.X = X
        self.Y = Y
        self.N = self.X.shape[0]
        self.sigma2 = sigma2  # sigma2 refers to noise variance
        self.kern = CovFunctions(covType, theta)
        self.K = self.kern.compute(X)
        # Force computations
        self.update_stats()

    def get_params(self):
        return np.hstack((self.sigma2, self.kern.get_theta()))

    def set_params(self, params):
        self.sigma2 = params[0]
        self.kern.set_theta(params[1:])

    def update_stats(self):
        self.K = self.kern.compute(self.X)
        self.Kinv = np.linalg.inv(self.K + self.sigma2 * np.eye(self.N))
        self.logdet = np.log(np.linalg.det(self.K + self.sigma2 * np.eye(self.N)))
        self.KinvY = np.dot(self.Kinv, self.Y)
        # Equivalent to: np.trace(np.dot(self.Y, self.KinvY.T))
        self.YKinvY = (self.Y * self.KinvY).sum()
        # self.YKinvY = np.dot(self.Y.T, self.KinvY)

    def likelihood(self):
        """
        That's actually the logarithm of equation (3)
        Since logarithm is a convex function, maximum likelihood and maximum log likelihood wrt parameters
        would yield the same solutuion, but logarithm is better to manage mathematically and
        numerically.
        """
        return -0.5 * (self.N * np.log(2 * np.pi) + self.logdet + self.YKinvY)

    def posterior(self, x_star):
        """
        Implements equation (4)
        """
        self.update_stats()
        K_starstar = self.kern.compute(x_star, x_star)  # (70,70)
        K_star = self.kern.compute(self.X, x_star)  # (22,70)
        KinvK_star = np.dot(self.Kinv, K_star)  # (22,70)
        mu_pred = np.dot(KinvK_star.T, self.Y)  # (70,1)
        K_pred = K_starstar - np.dot(KinvK_star.T, K_star)  # (70,70)
        return mu_pred, K_pred

    def objective(self, params):
        self.set_params(params)
        self.update_stats()
        return -self.likelihood()


# We'll create some sample 1D data as a draw from a GP with RBF covariance and lengthscale = 0.85.
# Next, we normalize the data and keep some for testing. We then plot the data.
N = 22  # number of training points
Nstar = 70  # number of test points

# create toy data
X = np.sort(np.random.rand(N+Nstar, 1) * 6, axis=0)  # (92,1)
K_rbf = cov_RBF(X, X, theta=np.array([1, 0.85]))  # (92, 92)
mu_temp = np.zeros((1, K_rbf.shape[0]))  # (1, 92)
mu = mu_temp[0, :]  # (92,)
#  mu = np.zeros(K_rbf.shape[0])

jitter = 1e-5 * np.eye(K_rbf.shape[0])
Y_temp = np.random.multivariate_normal(mean=mu, cov=K_rbf+jitter)  # (92,)
Y = Y_temp[:, None]  # (92, 1)
# Y = Y.reshape(-1, 1)

# split data into training and test set
perm = np.random.permutation(X.shape[0])  # perm, (92,)
Xtr = X[np.sort(perm[0:N], axis=0), :]  # (22, 1)
Ytr = Y[np.sort(perm[0:N], axis=0), :]
X_star = X[np.sort(perm[N:N+Nstar], axis=0), :]
Y_star = Y[np.sort(perm[N:N+Nstar], axis=0), :]

# Normalize data to be 0 mean and 1 std
Ymean = Y.mean()
Ystd = Y.std()
Ytr -= Ymean
Ytr /= Ystd
Y_star -= Ymean
Y_star /= Ystd

# plot data
plt.plot(X, (Y-Ymean)/Ystd, 'k-')
plt.plot(Xtr, Ytr, 'b<', label='training')
plt.plot(X_star, Y_star, 'ro', label='test')
plt.legend()
plt.grid()
plt.show()


# In general, we want a function that plots the fit showing the mean and variance:
def plot_fit(x, y, mu, var):
    """
    Plot the fit of a GP
    """
    plt.plot(x, y, 'k-o', label='true')
    plt.plot(x, mu, 'b-<', label='predicted')
    plt.plot(x, mu + 2 * np.sqrt(var), 'r--', label='var')
    plt.plot(x, mu - 2 * np.sqrt(var), 'r--')
    plt.legend()


# Now we'll define two GP models, one with linear and one with RBF cov. function.
# We'll use them to predict the training and the test data.
# Define GP models with initial parameters
g_lin = GP(Xtr, Ytr, 0.1, 'linear', 2)
g_rbf = GP(Xtr, Ytr, 0.1, 'RBF', np.array([1, 2]))

# Get the posterior of the two GPs on the *training* data
mu_lin_tr, var_lin_tr = g_lin.posterior(Xtr)
mu_rbf_tr, var_rbf_tr = g_rbf.posterior(Xtr)

# Get the posterior of the two GPs on the *test* data
mu_lin_star, var_lin_star = g_lin.posterior(X_star)
mu_rbf_star, var_rbf_star = g_rbf.posterior(X_star)
print('mu_rbf_star.shape', mu_rbf_star.shape)
print('var_rbf_star.shape', var_rbf_star.shape)

# Plot the fit of the two GPs on the training and test data
f = plt.figure(figsize=(20, 5))
ax = plt.subplot(2, 2, 1)
ax.set_aspect('auto')

temp_1 = np.diag(var_lin_tr)[:, None]
print('temp_1.shape', temp_1.shape)

plot_fit(Xtr, Ytr, mu_lin_tr, np.diag(var_lin_tr)[:, None])
plt.gca().set_title('Linear, training')

ax = plt.subplot(2, 2, 2)
ax.set_aspect('auto')
plot_fit(X_star, Y_star, mu_lin_star, np.diag(var_lin_star)[:, None])
plt.gca().set_title('Linear, test')

ax = plt.subplot(2, 2, 3)
ax.set_aspect('auto')
plot_fit(Xtr, Ytr, mu_rbf_tr, np.diag(var_rbf_tr)[:, None])
plt.gca().set_title('RBF (l=' + str(g_rbf.kern.get_theta()[1]) + '), training')

ax = plt.subplot(2, 2, 4)
ax.set_aspect('auto')
plot_fit(X_star, Y_star, mu_rbf_star, np.diag(var_rbf_star)[:, None])
plt.gca().set_title('RBF, test')

plt.show()


#  Even if we did not optimize the GPs, we see that both do a reasonably good job in fitting the data.
#  This is because the GP is a nonparametric model, in the sense that the data itself act as additional (constant) parameters.
#  Indeed, you can see that the posterior GP is given by an equation where the training data appear inside.
#  Let's convince ourselves that there is actually something to learn in the GP to get an even better fit.
#  In the folllowing, we'll test the fit of the RBF kernel for various selections of the lengthscale.
#  Remember that the true one generating the data is 0.85, but we'd expect to find it as the best only if we have infinite data.
#  Otherwise, something close to it is also expected.
#  In the next plot we show the fit for each lengthscale, and the Likelihood achieved by it.


f = plt.figure(figsize=(20, 28))
i = 1
# Following array holds different lengthscales to test one by one
test_l = [0.008, 0.05, 0.2, 0.85, 1.5, 2, 6, 12]
for l in test_l:
    g_rbf = GP(Xtr, Ytr, 0.1, 'RBF', np.array([1, l]))
    mu_rbf_tr, var_rbf_tr = g_rbf.posterior(Xtr)
    mu_rbf_star, var_rbf_star = g_rbf.posterior(X_star)

    ax = plt.subplot(len(test_l), 2, i)
    ax.set_aspect('auto')
    plot_fit(Xtr, Ytr, mu_rbf_tr, np.diag(var_rbf_tr)[:, None])
    ll = g_rbf.likelihood()
    plt.gca().set_title('RBF (l=' + str(g_rbf.kern.get_theta()[1]) + '), training. L=' + str(ll))

    ax = plt.subplot(len(test_l), 2, i+1)
    ax.set_aspect('auto')
    plot_fit(X_star, Y_star, mu_rbf_star, np.diag(var_rbf_star)[:, None])
    ll = g_rbf.likelihood()
    plt.gca().set_title('RBF (l=' + str(g_rbf.kern.get_theta()[1]) + '), test. L=' + str(ll))
    i += 2

plt.show()


# Fitting and Overfitting:
# We see that very short lenthscales give more "wiggly" functions,
# which are capable of interpolating perfectly between the training points.
# However, such an overfitted model is biased to be more "surprised" by anything else other than the exact same training data...
# hence performing poorly in the test set.
#
# On the other hand, high lengthscales give very flat functions, which in the limit look like the linear one and underfit the data.
# So, how do we automatically find the correct lengthscale?
#
# The principled way of selecting the correct parameter is not a visual check or testing the error in training/held out data.
# Instead, we wish to look at the likelihood, telling us what is the probability of the specific model (with lengthscale l) having generated the data.
#
# To show this, we'll try as before different settings for the lengthscale of the GP-RBF,
# but this time we'll keep track of the likelihood and also the training/test error.
#
# First, create some helper functions:


# Root mean squared error
def rmse(pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()
    return np.sqrt(np.mean((pred-truth) ** 2))


# Make data 0 mean and 1 std.
def standardize(x):
    return (x-x.mean())/x.std()


test_l = np.linspace(0.01, 5, 100)
ll = []
err_tr = []
err_test = []
for l in test_l:
    g_rbf = GP(Xtr, Ytr, 0.1, 'RBF', np.array([1, l]))
    g_rbf.update_stats()
    ll.append(g_rbf.likelihood())
    mu_rbf_tr, var_rbf_tr = g_rbf.posterior(Xtr)
    err_tr.append(rmse(mu_rbf_tr, Ytr))
    mu_rbf_star, var_rbf_star = g_rbf.posterior(X_star)
    err_test.append(rmse(mu_rbf_star, Y_star))
ll = standardize(np.array(ll))
err_tr = standardize(np.array(err_tr))
err_test = standardize(np.array(err_test))


# Plot the lenghtscale versus the likelihood, training error, test error:
max_ll, argmax_ll = np.max(ll), np.argmax(ll)
min_err_tr, argmin_err_tr = np.min(err_tr), np.argmin(err_tr)
min_err_test, argmin_err_test = np.min(err_test), np.argmin(err_test)

plt.plot(test_l, ll, label='likelihood')
plt.plot(test_l, err_tr, label='err_tr')
plt.plot(test_l, err_test, label='err_test')
tmp_x = np.ones(test_l.shape[0])
ylim = plt.gca().get_ylim()
tmp_y = np.linspace(ylim[0], ylim[1], tmp_x.shape[0])
plt.plot(tmp_x * test_l[argmax_ll], tmp_y, '--', label='maxL')
plt.plot(tmp_x * test_l[argmin_err_tr], tmp_y, '--', label='minErrTr')
plt.plot(tmp_x * test_l[argmin_err_test], tmp_y, '--', label='minErrTest')

plt.legend()
plt.show()

print('Best lengthscale according to likelihood    :' + str(test_l[argmax_ll]))
print('Best lengthscale according to training error:' + str(test_l[argmin_err_tr]))
print('Best lengthscale according to test error    :' + str(test_l[argmin_err_test]))


# The fact that the best lengthscale (according to likelihood) is not necessarily the one giving us less training error is a good property, i.e. the model avoids overfitting.
#
# The chosen lengthscale is also quite close to the etrue one. Running the experiment with more data will reaveal an even close match (bear in mind that numerical problems due to simple coding here might cause problems in computations).





































