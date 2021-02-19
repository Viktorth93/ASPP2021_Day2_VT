from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


mu = 10
sigma = 2

x = np.arange(norm.ppf(0.01,loc=mu,scale=sigma), norm.ppf(0.99,loc=mu, scale=sigma), 0.1)
print(x)
fig, [axpdf, axcdf, axhisto] = plt.subplots(1,3)


randVect = norm.rvs(loc=mu, scale=sigma, size=1000)

axpdf.plot(x,norm.pdf(x,mu), 'r-',label='PDF')
axpdf.legend(loc='best')
axcdf.plot(x,norm.cdf(x,mu), 'r-', label='CDF')
axcdf.legend(loc='best')
axhisto.hist(randVect, color='0.75', label='Normally distributed values')
axhisto.legend(loc='best')
fig.tight_layout()
plt.show()


