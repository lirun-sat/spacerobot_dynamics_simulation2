import matplotlib.pyplot as plt
import numpy as np


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


X = np.sort(np.random.rand(400, 1) * 6, axis=0)

params_linear = [0.01, 0.05, 1, 2, 4, 10]
params_rbf = [0.005, 0.1, 1, 2, 5, 12]
K = len(params_linear)


'''


plt.figure(figsize=(25, 25))
j = 1
for i in range(K):
    plt.subplot(K, 2, j)
    K_rbf = cov_RBF(X, X, theta=np.array([1, params_rbf[i]]))
    plt.imshow(K_rbf)
    plt.colorbar()
    plt.gca().set_title('RBF (l=' + str(params_rbf[i]) + ')')

    plt.subplot(K, 2, j + 1)
    K_lin = cov_linear(X, X, theta=params_linear[i])
    plt.imshow(K_lin)
    plt.colorbar()
    plt.gca().set_title('Lin (var=' + str(params_linear[i]) + ')')

    j += 2

plt.suptitle('RBF (left) and Linear (right) cov. matrices created with different parameters', fontsize=20)
plt.show()


'''

num_samples = 5
plt.figure(figsize=(25, 25))
j = 1
for i in range(K):
    plt.subplot(K, 2, j)
    K_rbf = cov_RBF(X, X, theta=np.array([1, params_rbf[i]]))
    plt.imshow(K_rbf)
    plt.colorbar()
    plt.gca().set_title('RBF Cov. Matrix (l=' + str(params_rbf[i]) + ')')

    plt.subplot(K, 2, j + 1)
    # Assume a GP with zero mean
    mu = np.zeros((1, K_rbf.shape[0]))[0, :]
    for s in range(num_samples):
        # Jitter is a small noise addition to the diagonal to ensure positive definiteness
        jitter = 1e-5 * np.eye(K_rbf.shape[0])
        sample = np.random.multivariate_normal(mean=mu, cov=K_rbf + jitter)
        plt.plot(sample)
    plt.gca().set_title('GP Samples from RBF (l=' + str(params_rbf[i]) + ')')
    j += 2
plt.show()























