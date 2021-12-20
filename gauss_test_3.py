
'''

# Plot the probability density function of a normal continuous random distribution

import scipy
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
w = 4
h = 3
d = 70
# plt.figure(figsize=(w, h), dpi=d)

a = norm.ppf(0.01)
print(a)
b = norm.ppf(0.99)
print(b)
x = np.linspace(a, b, 10)

# plt.scatter(x, norm.pdf(x))
plt.plot(x, norm.pdf(x))
plt.show()



# To find the PDF for a number x, call scipy.stats.norm.pdf(x, loc=None, scale=None) with loc set to the mean and scale to the standard deviation.
# To find the CDF for an interval from x to y, subtract scipy.stats.norm.cdf(x, loc=None, scale=None) from scipy.stats.norm.cdf(y, loc=None, scale=None) with loc set to the mean and scale to the standard deviation.

x = 1.0
pdf_probability = scipy.stats.norm.pdf(x, loc=0, scale=1)
print(pdf_probability)

y = 0.5
cdf_probability = scipy.stats.norm.cdf(x, loc=0, scale=1) - scipy.stats.norm.cdf(y, loc=0, scale=1)
print(cdf_probability)



# 要画出等高线，核心函数是plt.contourf()，但在这个函数中输入的参数是x,y对应的网格数据以及此网格对应的高度值，因此我们调用np.meshgrid(x,y)把x,y值转换成网格数据：
import numpy as np
import matplotlib.pyplot as plt

# 计算x,y坐标对应的高度值
def f(x, y):
    return (1 - x / 2 + x ** 3 + y ** 5) * np.exp(-x ** 2 - y ** 2)

# 生成x,y的数据
n = 9
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
print(type(x))

x_1 = np.array(x)
y_1 = np.array(y)
print(type(x_1))

# 把x,y数据生成mesh网格状的数据，因为等高线的显示是在网格的基础上添加上高度值
X, Y = np.meshgrid(x, y)
# X_1, Y_1 = np.mgrid[x, y]
a_1, b_1 = np.mgrid[-1:1:.01, -1:1:.01]

print(a_1)
print(b_1)

print(X)
print(Y)

pos = np.dstack((X, Y))
pos_1 = np.dstack((a_1, b_1))
print(pos)
print(pos_1)
# print(X_1)
# print(Y_1)

# 填充等高线
plt.contourf(X, Y, f(X, Y))
# 显示图表
plt.show()


'''


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import scipy as sp
from scipy.stats import multivariate_normal


def gen_Gaussian_samples(mu, sigma, N=200):
    """
    Generate N samples from a multivariate Gaussian with mean mu and covariance sigma
    """
    D = mu.shape[0]
    samples = np.zeros((N, D))
    for i in np.arange(N):
        samples[i, :] = np.random.multivariate_normal(mean=mu, cov=sigma)
    return samples.copy()


def gen_plot_Gaussian_samples(mu, sigma, N=1000):
    """
    Generate N samples from a multivariate Gaussian with mean mu and covariance sigma
    and plot the samples as they're generated
    """
    for i in np.arange(N):
        sample = np.random.multivariate_normal(mean=mu, cov=sigma)
        plt.plot(sample[0], sample[1], '.', color='r', alpha=0.6)
    plt.grid()


def plot_Gaussian_contours(x, y, mu, sigma, N=100):
    """
    Plot contours of a 2D multivariate Gaussian based on N points. Given points x and y are
    given for the limits of the contours
    """
    X, Y = np.meshgrid(np.linspace(x.min() - 0.3, x.max() + 0.3, 100), np.linspace(y.min() - 0.3, y.max() + 0.3, N))
    rv = multivariate_normal(mu, sigma)
    Z = rv.pdf(np.dstack((X, Y)))
    plt.contour(X, Y, Z)
    plt.xlabel('x_1')
    plt.ylabel('x_2')


def plot_sample_dimensions(samples, colors=None, markers=None, ms=10):
    """
    Given a set of samples from a bivariate Gaussian, plot them, but instead of plotting them
    x1 vs x2, plot them as [x1 x2] vs ['1', '2']
    """
    N = samples.shape[0]
    D = samples.shape[1]

    t = np.array(range(1, D + 1))

    for i in np.arange(N):
        if colors is None and markers is None:
            plt.plot(t, samples[i, :], '-o', ms=ms)
        elif colors is None:
            plt.plot(t, samples[i, :], '-o', marker=markers[i], ms=ms)
        elif markers is None:
            plt.plot(t, samples[i, :], '-o', color=colors[i], ms=ms)
        else:
            plt.plot(t, samples[i, :], '-o', color=colors[i], marker=markers[i], ms=ms)
    plt.grid()
    plt.xlim([0.8, t[-1] + 0.2])
    plt.ylim([samples.min() - 0.3, samples.max() + 0.3])
    plt.xlabel('d = {' + str(t) + '}')
    plt.ylabel('[x_d]')
    plt.gca().set_title(str(N) + ' samples from a bivariate Gaussian')


def set_limits(samples):
    plt.xlim([samples[:, 0].min() - 0.3, samples[:, 0].max() + 0.3])
    plt.ylim([samples[:, 1].min() - 0.3, samples[:, 1].max() + 0.3])


colors = ['r', 'g', 'b', 'm', 'k']
markers = ['p', 'd', 'o', 'v', '<']

N = 5  # Number of samples
mu = np.array([0, 0])  # Mean of the 2D Gaussian

sigmaUncor = np.array([[1, 0.02], [0.02, 1]])
sigmaCor = np.array([[1, 0.95], [0.95, 1]])

# But let's plot them as before dimension-wise...

samplesUncor = gen_Gaussian_samples(mu, sigmaUncor)
samplesCor = gen_Gaussian_samples(mu, sigmaCor)

f = plt.figure(figsize=(18, 5))
perm = np.random.permutation(samplesUncor.shape[0])[0::14]

ax1 = plt.subplot(1, 2, 1)
ax1.set_aspect('auto')
plot_sample_dimensions(samplesUncor[perm, :])
plt.gca().set_title('Weakly correlated')

ax2 = plt.subplot(1, 2, 2, sharey=ax1)
ax2.set_aspect('auto')

plot_sample_dimensions(samplesCor[perm, :])
plt.gca().set_title('Strongly correlated')
plt.ylim([samplesUncor.min()-0.3, samplesUncor.max()+0.3])

plt.show()
