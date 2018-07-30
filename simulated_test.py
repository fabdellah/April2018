# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:34:21 2018

@author: fabdellah
"""

import math
import numpy as np

class gas_storage(object):
    """ 
    S0 : initial spot price
    T : time to maturity 
    steps : number of discrete times (delta_t = T/steps)
    r : riskless discount rate (constant)
    sigma :  volatility of returns
    
    """

    def __init__(self,  S0,  T, steps,  r, sigma, nbr_simulations ):
     
            self.S0 = S0                                        # Parameters for the spot price
            self.T = T
            self.r = r
            self.sigma = sigma
            self.nbr_simulations = nbr_simulations
 
            
            if S0 < 0 or T <= 0 or r < 0  or sigma < 0 or nbr_simulations < 0 :
              raise ValueError('Error: Negative inputs not allowed')
 
            self.steps = steps  
            
            self.delta_t = self.T / float(self.steps)
            self.discount = np.exp(-self.r * self.delta_t)
                       
        
    def simulated_price_matrix(self, seed = 1):
        """ Returns Monte Carlo simulated prices (matrix)
            rows: time
            columns: price-path simulation """
   
        np.random.seed(seed)
        simulated_price_matrix = np.zeros((self.steps + 2, self.nbr_simulations), dtype=np.float64)
        simulated_price_matrix[0,:] = self.S0
        for t in range(1, self.steps + 2):
           # brownian = np.random.standard_normal( int(self.nbr_simulations / 2))
            brownian = np.random.standard_normal( 1)
           # brownian = np.concatenate((brownian, -brownian))        
            simulated_price_matrix[t, :] = (simulated_price_matrix[t - 1, :]      
                                  * np.exp((self.r - self.sigma ** 2 / 2.) * self.delta_t
                                  + self.sigma * brownian * np.sqrt(self.delta_t)))
            #needs to be specified according to the corresponding 2-factor model
        return simulated_price_matrix           


gas_storage1 = gas_storage(50, 1, 12,0.06, 0.3, 1)

simulated_prices = gas_storage1.simulated_price_matrix()



import matplotlib.pyplot as plt
time = np.arange(0,14)
plt.plot(time,gas_storage1.simulated_price_matrix()[:,:])   
plt.title('Simulated prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()



#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

import math
import numpy as np

_IGOLDEN = 2.0 / (math.sqrt(5.0) + 1.0)


def maximize_golden_section(
    func,
    output_shape,
    qranges,
    abs_tol=1e-4,
    max_iter=100
):
    """
    :param func: The function to be maximized. It must follow the form f(xs, values), and it evaluates xs and saves into values to avoid memory copy
    :param output_shape: the shape of output value grid
    :param qranges: shape=(n_x,2), representing (min, max) values for each x
    :param abs_tol: absolute error tolerance for x
    :param max_iter: Maximal iterations for the optimization
    :return: a structure of optimization results
    """
    
    print('maximize_golden_section')
    qa = np.zeros(shape=output_shape) + qranges[:, 0]
    qb = np.zeros(shape=output_shape) + qranges[:, 1]
    qabg = (qb - qa) * _IGOLDEN
    qc = qb - qabg
    fc = np.empty(shape=output_shape)
    func(qc, fc)
    qd = qa + qabg
    fd = np.empty(shape=output_shape)
    func(qd, fd)
    fe = np.empty(shape=output_shape)
    it = 0
    while np.max(np.abs(qc - qd)) > abs_tol and it < max_iter:
        is_c = fc > fd
        is_d = np.logical_not(is_c)
        qa[is_d] = qc[is_d]
        qb[is_c] = qd[is_c]
        qabg = (qb - qa) * _IGOLDEN
        qe = np.where(is_c, qb - qabg, qa + qabg)
        func(qe, fe)
        qd[is_c] = qe[is_c]
        qc[is_d] = qe[is_d]
        qc, qd = qd, qc
        fd[is_c] = fe[is_c]
        fc[is_d] = fe[is_d]
        fc, fd = fd, fc
        it += 1
    print('Total iterations:', it)
    success = it < max_iter
    q = (qc + qd) / 2.0
    func(q, fc)
    return {
        'x': q,
        'max_value': fc,
        'success': success,
        'iterations': it }


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def get_value_func(
        n_dim,
        states_grid,
        storage_levels,
        value_grid,
        price_func,
        reward_func,
        storage_level_func,
        storage_level_unit,):
    """
    :param n_dim: Dimension of price model states.
    :param states_grid: model state values at each grid coordinate. shape=[m1,m2,...mn,n], where n=n_dim. 
                        states_grid[i1,i2,...in,j] is the jth dimension of state value at coordinate (i1,i2,...in)
    :param storage_levels: 1-d grid saving the actual storage levels for each storage level index
    :param value_grid: storage value at each storage and model state. shape=[m1,m2,...,mn,mV], where mV=# of storage levels.
    :param price_func: takes states grid and returns prices for each state. Same shape as states_grid
    :param reward_func: the immediate reward or penalty following the injection/withdrawal action
    :param storage_level_func: returns updated storage levels given the injection/withdrawal action
    :param storage_level_unit: The volume represented by one unit of storage level index
    :return: the value function to be maximized
    """
    
    print('get_value_func')
    nq = len(storage_levels)
    nq0 = int(storage_levels[0] / storage_level_unit + 0.5)
    price_grid = price_func(states_grid)
    coords = [np.empty(0)] + [np.arange(value_grid.shape[i]) for i in range(n_dim)]

    def value_func(qs, grid):
        print('value_func')
        updated_storage_indices = storage_level_func(storage_levels, qs * storage_level_unit) / storage_level_unit
        lows = updated_storage_indices.astype(np.int)
        ps = updated_storage_indices - lows
        lows -= nq0
        highs = lows + 1
        lows = np.maximum(lows, 0)
        highs = np.minimum(highs, nq - 1)
        coords[0] = np.swapaxes(highs, 0, -1)
        high_values = np.swapaxes(np.swapaxes(value_grid, 0, -1)[coords], 0, -1)
        coords[0] = np.swapaxes(lows, 0, -1)
        low_values = np.swapaxes(np.swapaxes(value_grid, 0, -1)[coords], 0, -1)
        new_values = ps * high_values + (1.0 - ps) * low_values

        grid[:] = reward_func(price_grid, storage_levels, qs) + new_values

    return value_func


unit_ratio = 1.0
storage_level_unit = 50000000.0
max_storage_level_index = 100
target_storage_level_index = 50
unit_penalty = -storage_level_unit * max_storage_level_index * 100.0
inj_cost = 0.0
wth_cost = 0.0
trans_cost = 0.0

#n_dim = 1
n_dim = 2
max_state_index = 14
state_unit = 0.001
mean_price = 50.0


def price_func(states):
    print('price_func : ', np.exp(states[..., :1] ))
    print('length price_func : ', len( np.exp(states[..., :1] ) ))
    return np.exp(states[..., :1])


def reward_func(price_grid, storage_levels, q):
    print('reward_func')
    return -price_grid * q * unit_ratio * storage_level_unit - np.abs(q) * np.where(q > 0, inj_cost + trans_cost, wth_cost + trans_cost)


def storage_level_func(storage_levels, q):
    print('storage_level_func')
    return storage_levels + q


def final_value_func(states_grid, target):
    """
        Final value is 0 when final target is reached, and negative penalty when target is missed
    """
    print('final_value_func')
    grid = np.zeros(shape=list(states_grid.shape[:-1]) + [max_storage_level_index])
    target_idx_low = int(target / storage_level_unit)
    for i in range(max_storage_level_index):
        distance = max(target_idx_low - i, i - target_idx_low - 1)
        grid[..., i] = distance * unit_penalty
    return grid


def cartesian_product(*arrays):
    """ Fastest way to create cartesian coordinates
        See https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
        for benchmarks
    """
    print('cartesian_product')
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape([len(s) for s in arrays] + [la])

def create_coords(dx, i, n_dim):
    print('create_coords')
    coords = np.outer(dx, np.arange(0, 2 * i + 1) - i)
    print('shape(coords): ',coords.shape)
    print('coords : ', coords)
    return cartesian_product(*[coords[i] for i in range(n_dim)])


states_grid = create_coords(state_unit, int((max_state_index - 1) / 2), n_dim) + np.log(mean_price)
storage_indices = np.arange(max_storage_level_index)
storage_levels = storage_indices * storage_level_unit
value_grid = final_value_func(states_grid, storage_level_unit * target_storage_level_index)
storage_index_ranges = np.empty((max_storage_level_index, 2))
storage_index_ranges[:, 0] = np.maximum(-10.0, -storage_indices).astype(np.int)
storage_index_ranges[:, 1] = np.minimum(15.0, max_storage_level_index - storage_indices - 1.0).astype(np.int)

output_shape = list(states_grid.shape[:-1]) + [len(storage_levels)]
print('output shape of max golden optimization: ', output_shape)
value_func = get_value_func(
    n_dim,
    states_grid,
    storage_levels,
    value_grid,
    price_func,
    reward_func,
    storage_level_func,
    storage_level_unit,
)
ret = maximize_golden_section(
    value_func,
    output_shape,
    storage_index_ranges,
    abs_tol=1e-4,
    max_iter=100
)
optimized_control = ret['x']
best_value_grid = ret['max_value']
print('Done')




    
    
    
    
    
    
    
    


