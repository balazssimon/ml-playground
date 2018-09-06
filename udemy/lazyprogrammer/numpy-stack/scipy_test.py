import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

norm.pdf(0)

# mean: loc, stddev: scale
norm.pdf(0, loc=5, scale=10)

r = np.random.randn(10)

# probability distribution function:
norm.pdf(r)

# log probability:
norm.logpdf(r)

# cumulative distribution function:
norm.cdf(r)

# log cumulative distribution function:
norm.logcdf(r)

# sampling from standard normal:
r = np.random.randn(10000)
plt.hist(r, bins=100)
plt.show()

mean = 5
stddev = 10
r = stddev*np.random.randn(10000)+mean
plt.hist(r, bins=100)
plt.show()

# spherical Gaussian:
r = np.random.randn(10000, 2)
plt.scatter(r[:,0], r[:,1])
plt.show()

# elliptical Gaussian:
r[:,1] = stddev*r[:,1]+mean
plt.scatter(r[:,0], r[:,1])
plt.axis('equal')
plt.show()

# non-axis-aligned Gaussian:
cov = np.array([[1,0.8],[0.8,3]]) # covariant matrix, covariance: 0.8 in both dimensions
mu = np.array([0,2])
r = mvn.rvs(mean=mu, cov=cov, size=1000)
plt.scatter(r[:,0], r[:,1])
plt.axis('equal')
plt.show()

r = np.random.multivariate_normal(mean=mu, cov=cov, size=1000)
plt.scatter(r[:,0], r[:,1])
plt.axis('equal')
plt.show()

# loading Matlab .mat files: scipy.io.loadmat

# loading WAV files: scipy.io.wavfile.read (default sampling: 44100 Hz)
# saving WAV files: scipy.io.wavfile.write

# signal processing, filtering: scipy.signal
# convolution: scipy.signal.convolve, scipy.signal.convolve2d

# FFT is in numpy:
x = np.linspace(0, 100, 10000)
y = np.sin(x) + np.sin(3*x) + np.sin(5*x)
plt.plot(x, y)
plt.show()

Y = np.fft.fft(y)
plt.plot(np.abs(Y))
plt.show()

2*np.pi*16/100
2*np.pi*48/100
2*np.pi*80/100
