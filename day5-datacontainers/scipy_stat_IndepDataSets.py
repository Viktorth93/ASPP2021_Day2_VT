from scipy.stats import norm
from scipy.stats import crystalball
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt

# Generate data from two different normal distributions


randVect1 = norm.rvs(loc=5, scale=2, size=1000000)
#randVect2 = norm.rvs(loc=5.5, scale=10, size=10000)
randVect2 = crystalball.rvs(beta=1, m=1.5, loc=5, scale=2, size=1000000)

[stat, pval] = ttest_ind(randVect1, randVect2)

print("P-Value: ", pval)

if pval <0.05: 
    print("We reject the null hypothesis that the datasets come from the same distribution at 95% confidence level")
else:
    print("We do not reject the null hypothesis that the datasets come from the same distribution at 95% confidence level")


fig, axpdf = plt.subplots(1,1)

bins = np.linspace(-30,50,100)

axpdf.hist(randVect1, bins, alpha=0.5, color='k', label='Gauss')
axpdf.hist(randVect2, bins, alpha =0.5, color='r', label='CB')
axpdf.legend(loc='best')
fig.tight_layout()
plt.show()
