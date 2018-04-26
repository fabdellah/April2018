# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:08:30 2018

@author: fabdellah
"""



class Schwartz_calibration(object):
       
    def __init__(self, theta, sigma_market):
            self.theta = theta
            self.sigma_market = sigma_market

    def compute_parameters(self,forward_market):
        t = np.arange(1,len(forward_market))
        mu_0 = forward_market[0]
        forward_market_t = forward_market[1:len(forward_market)]
        mu = np.divide( (mu_0*np.exp(-self.theta*t)-np.log(forward_market_t)) , np.exp(-self.theta*t)-1 ) 
        sigma = np.divide( 2*self.theta * self.sigma_market , 1-np.exp(-2*self.theta*t) )
        return mu,sigma


schwartz = Schwartz_calibration(1 , 0.4)  #theta, sigma_market
mu , sigma = schwartz.compute_parameters(forward_market)
    



fwd_curve = instruments['forward_curve']
ts = fwd_curve[:, 0]
ethetats = np.exp(ts * theta)

prices = fwd_curve[:, 1]
vrs = sigma * sigma / theta / 2.0 * (1.0 - 1.0 / (ethetats * ethetats))
mius = np.log(prices) - vrs / 2.0

# Calibrate miu means.
dethetats = np.diff(ethetats)
miumeans = []
for i in range(1, prices.shape[0]):
   miumeans.append((mius[i] * ethetats[i] - mius[0] - np.inner(dethetats[:i - 1], miumeans)) / (dethetats[i - 1]))

The model: d miu = theta(miu_mean(t) - miu)dt + sigma dW





