# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:46:57 2018

@author: fabdellah
"""


import math
import numpy as np
from numpy import random

def payoff1(price ,dv, inj_cost , wth_cost, trans_cost):  
      h=  -price*dv*np.where(dv > 0, inj_cost + trans_cost, wth_cost + trans_cost)     
      return h



def payoff(S,dv,injection_cost,withdrawal_cost):    
      return -withdrawal_cost*dv*S


def penalty(S,v):
    
    #specify the penalty function otherwise return 0
    return 0
    



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
        'iterations': it
    }


class gas_storage(object):
    """ 
    S0 : initial spot price
    T : time to maturity 
    steps : number of discrete times (delta_t = T/steps)
    r : riskless discount rate (constant)
    sigma :  volatility of returns
    
    """

    def __init__(self,  S0,  T, steps,M,  r, sigma, nbr_simulations,vMax,vMin , max_rate, min_rate , injection_cost, withdrawal_cost ):
     
            self.S0 = S0                                        # Parameters for the spot price
            self.T = T
            self.r = r
            self.sigma = sigma
            self.nbr_simulations = nbr_simulations       
        
            self.vMax = vMax                                    # Parameters for facility
            self.vMin = vMin                               
            self.max_rate = max_rate             
            self.min_rate = min_rate             
            
            self.injection_cost =injection_cost                 # Parameters for the payoff function
            self.withdrawal_cost = withdrawal_cost

            
            if S0 < 0 or T <= 0 or r < 0  or sigma < 0 or nbr_simulations < 0 :
              raise ValueError('Error: Negative inputs not allowed')
 
            self.steps = steps  
            self.M = M
            self.alpha = self.vMax/(self.M-1)
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
            brownian = np.random.standard_normal( int(self.nbr_simulations / 2))
            brownian = np.concatenate((brownian, -brownian))        
            simulated_price_matrix[t, :] = (simulated_price_matrix[t - 1, :]      
                                  * np.exp((self.r - self.sigma ** 2 / 2.) * self.delta_t
                                  + self.sigma * brownian * np.sqrt(self.delta_t)))
            #needs to be specified according to the corresponding 2-factor model
        return simulated_price_matrix           


    
    def contract_value(self):
    
        
        value_matrix = np.zeros((self.simulated_price_matrix().shape[0],self.simulated_price_matrix().shape[1],self.M))  # time, path , volume level
        acc_cashflows = np.zeros_like(value_matrix)
        decision_rule = np.zeros_like(value_matrix)
        volume_level = np.zeros_like(self.simulated_price_matrix())
        #volume_level_stored = np.zeros_like(self.simulated_price_matrix())
        
        decision_rule_avg = np.zeros((self.simulated_price_matrix().shape[0],self.M))
        volume_level_avg = np.zeros(self.simulated_price_matrix().shape[0])
        acc_cashflows_avg = np.zeros((self.simulated_price_matrix().shape[0],self.M))
        #volume_level_avg[0] = self.V0
        #volume_level_avg[-1] = self.V_end
        
        #volume_level_stored = np.zeros_like(self.simulated_price_matrix())[0, : ] = self.V0*np.ones(self.nbr_simulations)
        #volume_level_stored[-1,: ] = self.V_end*np.ones(self.nbr_simulations)
        value_matrix[-1,: ,:] = penalty(self.simulated_price_matrix()[-1, :],volume_level[-1,:]) 
        acc_cashflows[-1,:,:] = penalty(self.simulated_price_matrix()[-1, :],volume_level[-1,:])
        
        
        
        from scipy.optimize import minimize
        
        
        for t in range(self.steps , 0 , -1):
        
            print ('-----------')
            print ('Time: %5.3f, Spot price: %5.1f ' % (t, self.simulated_price_matrix()[t, 1]))           
            for m in range(1,self.M):    
                
                volume_level[t+1,:] = (m-1)*self.alpha

                regression = np.polyfit(self.simulated_price_matrix()[t, :], acc_cashflows[t+1, :, m-1] * self.discount, 2)
                continuation_value = np.polyval(regression, self.simulated_price_matrix()[t, :])
                
                for b in range(self.nbr_simulations):
                     f = lambda x: -1*( payoff(self.simulated_price_matrix()[t, b],
                                               x ,self.injection_cost,self.withdrawal_cost ) + continuation_value[b]  )
    
                     cons = ({'type': 'ineq', 'fun': lambda x:  (volume_level[t+1,b] - x - self.vMin)        },   
                             {'type': 'ineq', 'fun': lambda x:  (self.vMax - volume_level[t+1,b] + x)        },
                             {'type': 'ineq', 'fun': lambda x:  (self.max_rate - x)                          },
                             {'type': 'ineq', 'fun': lambda x:  (x - self.min_rate)                          },
                             {'type': 'ineq', 'fun': lambda x:  (self.max_rate*t - volume_level[t+1,b] + x ) })   
        
                     res = minimize(f, random.rand(1), constraints=cons)     
                     decision_rule[t,b,m-1] = res.x
    
                   
                   
                acc_cashflows[t,:,m-1] = payoff(self.simulated_price_matrix()[t, :],
                             decision_rule[t,:,m-1] , self.injection_cost, self.withdrawal_cost) + acc_cashflows[t+1,:,m-1]*self.discount
                
                decision_rule_avg[t,m-1] = np.sum(decision_rule[t,:,m-1])/self.nbr_simulations
                volume_level_avg[t] = np.sum(volume_level[t,:])/self.nbr_simulations
                acc_cashflows_avg[t,m-1] = np.sum(acc_cashflows[t,:,m-1])/self.nbr_simulations
                
                #print (' Decision rule (inject/withdraw): %5.1f , Acc_cashflows: %5.2f , payoff: %5.2f  '
                     # % (decision_rule[t,1,m-1], acc_cashflows[t,1,m-1],  payoff(self.simulated_price_matrix()[t, 1],decision_rule[t,1,m-1] , self.injection_cost, self.withdrawal_cost)))
                     
          
        contract_value = acc_cashflows[1,:,:] * self.discount             # at time 0
        
        return contract_value,acc_cashflows,decision_rule


    
    def price(self):
        return round((np.sum(self.contract_value()[0],axis=0) /  float(self.nbr_simulations))[0] , 2)
            
            
    
# gas_storage(S0,  T, steps, M, r, sigma, nbr_simulations, vMax, vMin , max_rate , min_rate  , injection_cost, withdrawal_cost )  
         
#facility1 =  gas_storage(3, 0 , 0, 1, 12, 11, 0.06, 0.3, 50, 5000000, 0 , 34000, -53000  ,0.1, 0.1)  
  
facility1 =  gas_storage(6, 1, 12, 101, 0.06, 0.3, 50, 5000000000, 0 , 35000, -75000  ,0.01, 0.01)  

contract_value, acc_cashflows, decision_rule = facility1.contract_value()

steps = 12
nbr_simulations = 50
alpha = 2500
M = 101
V_end=2500
volume_level_stored = np.zeros((steps+2, nbr_simulations))
volume_level_stored[steps,:] = V_end

for t  in range(steps,0,-1):
    
    for b in range(nbr_simulations):
        
        for m in range(1,M):
            
            if (volume_level_stored[t+1,b]==(m-1)*alpha):
                
                if (decision_rule[t,b,m] != 0):   
                        
                       volume_level_stored[t,b] = volume_level_stored[t+1,b] - (np.sign(decision_rule[t,1,m]))*((abs(decision_rule[t,1,m])//alpha)+1)*alpha 
                else:
                       volume_level_stored[t,b] = volume_level_stored[t+1,b]
        
    
#to delete

for t  in range(steps,0,-1):
    
    for b in range(nbr_simulations):
        
        for m in range(1,M):
            if (volume_level_stored[t+1,b]==(m-1)*alpha):
                
                if (contract_value[t,b,m] != 0):   
                        
                       volume_level_stored[t,b] = volume_level_stored[t+1,b] - (np.sign(contract_value[t,1,m]))*((abs(contract_value[t,1,m])//alpha)+1)*alpha 
                else:
                       volume_level_stored[t,b] = volume_level_stored[t+1,b]
             
                  
#to delete


from time import time
t0 = time()
print ('Price: ' ,facility1.price() )   
t1 = time();
d1 = t1 - t0
print ("Duration in Seconds %6.3f" % d1)       


import matplotlib.pyplot as plt
time = np.arange(0,14)
plt.plot(time,facility1.simulated_price_matrix()[:,:])   
plt.title('Simulated prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

import matplotlib.pyplot as plt
time = np.arange(0,self.steps)
plt.plot(time,acc_cashflows[0:12,:,1])   
plt.title('Acc_cashflows')
plt.xlabel('Time')
plt.ylabel('Acc_cashflows [$]')
plt.show()

