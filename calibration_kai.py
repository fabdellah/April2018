# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:00:25 2018

@author: fabdellah
"""

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