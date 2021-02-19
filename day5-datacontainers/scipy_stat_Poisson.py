from scipy.stats import poisson
import numpy as np
import matplotlib.pyplot as plt


mu = 4

x = np.arange(poisson.ppf(0.1,mu), poisson.ppf(0.9,mu))
fig, [axpmf, axcdf, axhisto] = plt.subplots(1,3)


randVect = poisson.rvs(mu, size=1000)

axpmf.plot(x,poisson.pmf(x,mu), 'bo',label='poisson pmf')
axpmf.vlines(x,0, poisson.pmf(x,mu), colors='b')
axpmf.legend(loc='best')
axcdf.plot(x,poisson.cdf(x,mu), 'bo', label='poisson cdf')
axcdf.legend(loc='best')
axhisto.hist(randVect, color='0.75', label='Poisson distributed values')
axhisto.legend(loc='best')
fig.tight_layout()
plt.show()


