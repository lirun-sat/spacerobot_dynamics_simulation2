import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def gaussian_kernel(x1, x2, l=1.0, sigma_f=1.0):
    """Easy to understand but inefficient."""
    m, n = x1.shape[0], x2.shape[0]
    dist_matrix = np.zeros((m, n), dtype=float)
    for i in range(m):
        for j in range(n):
            dist_matrix[i][j] = np.sum((x1[i] - x2[j]) ** 2)
    return sigma_f ** 2 * np.exp(- 0.5 / l ** 2 * dist_matrix)


def gaussian_kernel_vectorization(x1, x2, leng=1.0, sigma_f=1.0):
    """More efficient approach."""
    temp_1 = np.sum(x1 ** 2, 1).reshape(-1, 1)
    # temp_2 = np.sum(x2 ** 2, 1)
    temp_2 = np.sum(x2 ** 2, 1).reshape(1, -1)
    temp_3 = temp_1 + temp_2
    dist_matrix = temp_3 - 2 * np.dot(x1, x2.T)
    # dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1).reshape(-1, 1) - 2 * np.dot(x1, x2.T)
    return sigma_f ** 2 * np.exp(-0.5 / leng ** 2 * dist_matrix)


x = np.array([700, 800, 1029]).reshape(-1, 1)

print(gaussian_kernel_vectorization(x, x, leng=500, sigma_f=10))


class GPR:

    def __init__(self, optimize=True):
        self.is_fit = False
        self.train_X, self.train_y = None, None
        self.params = {"l": 0.5, "sigma_f": 0.2}
        self.optimize = optimize

    def fit(self, X, y):
        # store train data
        self.train_X = np.asarray(X)
        self.train_y = np.asarray(y)

        # hyper parameters optimization
        def negative_log_likelihood_loss(params):
            self.params["l"], self.params["sigma_f"] = params[0], params[1]
            Kyy = self.kernel(self.train_X, self.train_X) + 1e-8 * np.eye(len(self.train_X))
            temp = 0.5 * self.train_y.T.dot(np.linalg.inv(Kyy)).dot(self.train_y) + 0.5 * np.linalg.slogdet(Kyy)[1] + 0.5 * len(self.train_X) * np.log(2 * np.pi)
            # print(temp.shape)
            temp_1 = temp.flatten()
            return temp_1

        if self.optimize:
            res = minimize(negative_log_likelihood_loss, np.array([self.params['l'], self.params["sigma_f"]]),
                           bounds=((1e-4, 1e4), (1e-4, 1e4)),
                           method='L-BFGS-B')
            self.params["l"], self.params["sigma_f"] = res.x[0], res.x[1]

        self.is_fit = True

    def predict(self, X):
        if not self.is_fit:
            print("GPR Model not fit yet.")
            return

        X = np.asarray(X)
        Kff = self.kernel(self.train_X, self.train_X)  # (N, N)
        Kyy = self.kernel(X, X)  # (k, k)
        Kfy = self.kernel(self.train_X, X)  # (N, k)
        Kff_inv = np.linalg.inv(Kff + 1e-8 * np.eye(len(self.train_X)))  # (N, N)

        mu = Kfy.T.dot(Kff_inv).dot(self.train_y)
        cov = Kyy - Kfy.T.dot(Kff_inv).dot(Kfy)
        return mu, cov

    def kernel(self, x1, x2):
        dist_matrix = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
        return self.params["sigma_f"] ** 2 * np.exp(-0.5 / self.params["l"] ** 2 * dist_matrix)


def y_2d(x, noise_sigma=0.0):
    x = np.asarray(x)
    y = np.sin(0.5 * np.linalg.norm(x, axis=1))
    y += np.random.normal(0, noise_sigma, size=y.shape)
    return y


train_X = np.random.uniform(-4, 4, (100, 2)).tolist()
train_y = y_2d(train_X, noise_sigma=1e-4)

test_d1 = np.arange(-5, 5, 0.2)
test_d2 = np.arange(-5, 5, 0.2)
test_d1, test_d2 = np.meshgrid(test_d1, test_d2)
test_X = [[d1, d2] for d1, d2 in zip(test_d1.ravel(), test_d2.ravel())]

gpr = GPR(optimize=True)
gpr.fit(train_X, train_y)
mu, cov = gpr.predict(test_X)
z = mu.reshape(test_d1.shape)

fig = plt.figure(figsize=(7, 5))
ax = Axes3D(fig)
ax.plot_surface(test_d1, test_d2, z, cmap=cm.coolwarm, linewidth=0, alpha=0.2, antialiased=False)
ax.scatter(np.asarray(train_X)[:, 0], np.asarray(train_X)[:, 1], train_y, c=train_y, cmap=cm.coolwarm)
ax.contourf(test_d1, test_d2, z, zdir='z', offset=0, cmap=cm.coolwarm, alpha=0.6)
ax.set_title("l=%.2f sigma_f=%.2f" % (gpr.params["l"], gpr.params["sigma_f"]))

plt.show()













