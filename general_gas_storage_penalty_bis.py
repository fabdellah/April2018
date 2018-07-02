# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:51:12 2018

@author: fabdellah
"""

import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
from datetime import timedelta, date, datetime
from scipy.optimize import differential_evolution
from dateutil.relativedelta import relativedelta
from scipy import optimize
from numpy import linalg as LA
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import RidgeCV
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', np.RankWarning)
from time import time
import math
from sklearn import datasets, linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 
from scipy.optimize import minimize
from numpy import random
import seaborn as sns




    

 


class gas_storage(object):
    """ 
    simulated_price_matrix_fwd : simulated gas prices from the predicted prices 
    T : time to maturity   
    r : riskless discount rate (constant)
    sigma_GBM :  volatility of prices for simulating Geometric brownian motion prices
    nbr_simulation : number of simulated paths
    
    vMax : maximum capacity [MWh]
    vMin : minimum capacity [MWh]
    max_rate : maximum injection rate [MWh/day]
    min_rate : maximum withdrawal rate [MWh/day]
    
    injection_cost :cost of injection [EUR/MWh]
    withdrawal_cost : cost of withdrawal [EUR/MWh]
    
    steps : number of discrete times (delta_t = T/steps)
    M : number of units of the maximum capacity  
    alpha : fixed width of the discretized volume
    
    v_start : Volume at the starting time

    """

    def payoff(self,S, dv, injection_cost, withdrawal_cost):   
        """specify the payoff."""
        return S*dv*np.where(dv > 0, injection_cost, -withdrawal_cost )     
     
    
    def penalty1(self,S,v,v_target):    
        """specify the penalty function."""
        if (v!=v_target):
            penalty = -0.1
        else:
            penalty = 0
        return penalty
    
    def penalty(self):    
        """specify the penalty function (set to zero for simplicity)."""
        return 0   
 

    def __init__(self, simulated_price_matrix_fwd, T, steps, M, r, sigma_GBM, nbr_simulations, vMax, vMin, inj_rate, with_rate, injection_cost, withdrawal_cost, v_start, v_end):
     
            self.simulated_price_matrix_fwd = simulated_price_matrix_fwd                                       
            self.T = T
            self.r = r
            self.sigma_GBM = sigma_GBM
            self.nbr_simulations = nbr_simulations       
        
            self.vMax = vMax                                    
            self.vMin = vMin                               
            self.inj_rate = inj_rate             
            self.with_rate = with_rate             
            
            self.injection_cost =injection_cost                 
            self.withdrawal_cost = withdrawal_cost
        
            if T <= 0 or r < 0  or sigma_GBM < 0 or nbr_simulations < 0 :
              raise ValueError('Error: Negative inputs not allowed')
 
            self.steps = steps  
            self.M = M
            self.alpha = self.vMax/(self.M-1)
            self.delta_t = self.T / float(self.steps)
            self.discount = np.exp(-self.r * self.delta_t)
              
            self.v_start = v_start
            self.v_end = v_end
            
            
    def max_inj_rates(self,vol=1):
        #return min(self.vMax - vol , self.inj_rate )
        return self.inj_rate

    def max_wit_rates(self,vol=1):
        #return max(self.vMin - vol ,self.with_rate )
        return self.with_rate
        
        
    def simulated_price_matrix_GBM(self, seed = 1):
        """ Returns Monte Carlo simulated prices (matrix) with a Geometric Brownian Motion
            rows: time
            columns: price-path simulation """
    
        np.random.seed(seed)
        simulated_price_matrix_GBM = np.zeros((self.steps + 2, self.nbr_simulations), dtype=np.float64)
        S0 = 3
        simulated_price_matrix_GBM[0,:] = S0
        for t in range(1, self.steps + 2):
            brownian = np.random.standard_normal( int(self.nbr_simulations / 2))
            brownian = np.concatenate((brownian, -brownian))        
            simulated_price_matrix_GBM[t, :] = (simulated_price_matrix_GBM[t - 1, :]      
                                  * np.exp((self.r - self.sigma_GBM ** 2 / 2.) * self.delta_t
                                  + self.sigma_GBM * brownian * np.sqrt(self.delta_t)))
            #needs to be specified according to the corresponding 2-factor model
        return simulated_price_matrix_GBM           
    

    def simulated_price_matrix(self):
        """ Returns a contango curve (just for testing the model) """
        simulated_price_matrix = np.zeros((self.steps + 2, self.nbr_simulations), dtype=np.float64)
        for b in range(0, self.nbr_simulations):
            simulated_price_matrix[:,b] = np.linspace(1,8,14)
        return simulated_price_matrix
      

    
    def contract_value(self):
        """Returns the value of the contract at time 0, the accumulated cashflows, the decision rule and the final price of the storage."""   
        days_per_month = 31
       # value_matrix = np.zeros((self.simulated_price_matrix().shape[0],self.simulated_price_matrix().shape[1],self.M))  # time, path , volume level
        acc_cashflows = np.zeros((self.simulated_price_matrix().shape[0],self.simulated_price_matrix().shape[1],self.M))  # time, path , volume level
        
        decision_rule = np.zeros((self.simulated_price_matrix().shape[0],self.simulated_price_matrix().shape[1],self.M))  # time, path , volume level
        volume_level = np.zeros_like(self.simulated_price_matrix())
        
        decision_rule_avg = np.zeros((self.simulated_price_matrix().shape[0],self.M))
        volume_level_avg = np.zeros(self.simulated_price_matrix().shape[0])
        acc_cashflows_avg = np.zeros((self.simulated_price_matrix().shape[0],self.M))
       # value_matrix[-1,: ,:] = self.penalty()
        acc_cashflows[-1,:,:] = self.penalty()
               
        for t in range(self.steps , 0 , -1):
        
            print ('-----------')
            print ('Time: %5.3f, Spot price: %5.1f ' % (t, self.simulated_price_matrix()[t, 1]))           
            for m in range(1,self.M):    
                
                volume_level[t+1,:] = (m-1)*self.alpha

                regression = np.polyfit(self.simulated_price_matrix()[t, :], acc_cashflows[t+1, :, m-1] * self.discount, 2)
                continuation_value = np.polyval(regression, self.simulated_price_matrix()[t, :])
                
                acc_delta_v = 0
                
                for b in range(self.nbr_simulations):
                     f = lambda x: -1*( self.payoff(self.simulated_price_matrix()[t, b],
                                               x ,self.injection_cost,self.withdrawal_cost ) + continuation_value[b]  )
#    
#                     cons = ({'type': 'ineq', 'fun': lambda x:  (volume_level[t+1,b] - x - self.vMin)            },   
#                             {'type': 'ineq', 'fun': lambda x:  (self.vMax - volume_level[t+1,b] + x)            },
#                             
#                             {'type': 'ineq', 'fun': lambda x:  (self.max_inj_rates(volume_level[t+1,b]-x) - x)  },
#                             {'type': 'ineq', 'fun': lambda x:  (x - self.max_wit_rates(volume_level[t+1,b]-x))  }, 
#                                            
#                             {'type': 'ineq', 'fun': lambda x:  (self.v_start + self.max_inj_rates(volume_level[t+1,b]-x)*t - volume_level[t+1,b] + x ) })   
                     
                     cons = ({'type': 'ineq', 'fun': lambda x:  (days_per_month*self.max_inj_rates() - x)             },   
                             {'type': 'ineq', 'fun': lambda x:  (x - days_per_month*self.max_wit_rates() )             },  
                             
                             
                             {'type': 'ineq', 'fun': lambda x:  (self.vMax - self.v_end + ( acc_delta_v + x ))  },
                             {'type': 'ineq', 'fun': lambda x:  (self.v_end - ( acc_delta_v + x ) - self.vMin)  }  ,
                             
                             {'type': 'eq', 'fun': lambda x:    (self.v_end - self.v_start - (acc_delta_v + x ) ) })
                             
             
                     res = minimize(f, [random.rand(1)], constraints=cons)     
                     decision_rule[t,b,m-1] = res.x       
                     acc_delta_v = acc_delta_v + decision_rule[t,b,m-1]
                   
                acc_cashflows[t,:,m-1] = self.payoff(self.simulated_price_matrix()[t, :],
                             decision_rule[t,:,m-1] , self.injection_cost, self.withdrawal_cost) + acc_cashflows[t+1,:,m-1]*self.discount
                
                decision_rule_avg[t,m-1] = np.sum(decision_rule[t,:,m-1])/self.nbr_simulations
                volume_level_avg[t] = np.sum(volume_level[t,:])/self.nbr_simulations
                acc_cashflows_avg[t,m-1] = np.sum(acc_cashflows[t,:,m-1])/self.nbr_simulations
          
        contract_value = acc_cashflows[1,:,:] * self.discount             # at time 0
        
        price = round((np.sum(contract_value,axis=0) /  float(self.nbr_simulations))[0] , 2)
        
        return contract_value, acc_cashflows, decision_rule, price

 
            



def volume_level_func(decision_rule, steps, nbr_simulations, volume_init, alpha,specification):
    """Returns the optimal volume stored at each time given a volume at the final time""" 
    decision_rule = np.around(decision_rule, decimals=-1)
    volume_level_stored = np.zeros((steps+2, nbr_simulations))
    
    for b in range(nbr_simulations):  
        if (specification=='end'):     
            volume_level_stored[steps+1,:] = volume_init
            for t  in range(steps,0,-1):     
                m_level = ( (volume_level_stored[t+1,b])/alpha).astype(int) + 1     
                if (np.abs(decision_rule[t,b,m_level]) >= alpha):                   
                       volume_level_stored[t,b] = volume_level_stored[t+1,b] - decision_rule[t,b,m_level]
                else:
                       volume_level_stored[t,b] = volume_level_stored[t+1,b]
        elif (specification=='start'):
            volume_level_stored[0,:] = volume_init
            for t  in range(steps):   
                m_level = ( (volume_level_stored[t,b])/alpha).astype(int) + 1
                if (np.abs(decision_rule[t,b,m_level]) >= alpha):                   
                       volume_level_stored[t+1,b] = volume_level_stored[t,b] + decision_rule[t,b,m_level]
                else:
                       volume_level_stored[t+1,b] = volume_level_stored[t,b]
               
    return volume_level_stored


def decision_volume(decision_rule, steps, nbr_simulations, volume_init, alpha, b_example, specification):
    """Returns the optimal behaviour (inject or withdraw) at each time given a volume at the final time"""
    inj = np.zeros(steps+2)
    withd = np.zeros(steps+2)
    decision_rule = np.around(decision_rule, decimals=-1)
    volume_level_stored = volume_level_func(decision_rule, steps, nbr_simulations, volume_init, alpha, specification) 
    volume_level_stored_ex =volume_level_stored[:,b_example]  
    if (specification=='end'): 
        for t  in range(steps,0,-1):                
                m_level = ( (volume_level_stored_ex[t+1])/alpha).astype(int) + 1    
                print('At time ', t , ', inject/withdraw: ',decision_rule[t,b_example,m_level] )           
                if decision_rule[t,b_example,m_level]>0 :
                    inj[t] = decision_rule[t,b_example,m_level]
                else:
                    withd[t] = np.abs(decision_rule[t, b_example, m_level])
    elif (specification=='start'):
         for t  in range(steps):                
                m_level = ( (volume_level_stored_ex[t])/alpha).astype(int) + 1    
                print('At time ', t , ', inject/withdraw: ',decision_rule[t,b_example,m_level] )           
                if decision_rule[t,b_example,m_level]>0 :
                    inj[t] = decision_rule[t,b_example,m_level]
                else:
                    withd[t] = np.abs(decision_rule[t, b_example, m_level])
    return withd, inj





file = 'simulated_price_matrix.xlsx'            #not used for now
simulated_price_matrix = pd.read_excel(file)
simulated_price_matrix = simulated_price_matrix.values
    


facility =  gas_storage(simulated_price_matrix, 1, 12, 101, 0.035, 0.1, 2, 1000000, 0 , 25000, 75000  ,0.1, 0.1, 0, 0)  
contract_value, acc_cashflows, decision_rule, price = facility.contract_value() 



M = 101
nbr_simulations = 2
alpha = 2500
volume_end = 0
volume_start = 0
steps = 12

volume_level = volume_level_func(decision_rule, steps, nbr_simulations, volume_end, alpha, 'end')
withd, inj = decision_volume(decision_rule, steps, nbr_simulations, volume_end, alpha, 0, 'end')





time = np.arange(0,14)  
plt.plot(time, facility.simulated_price_matrix()  ) 
plt.title('Contango')
plt.xlabel('Time (month)')
plt.ylabel('Price [EUR/MWh]')
plt.show()


# Plot the optimal volume stored (example)
time = np.arange(0,14)
plt.rcParams['figure.figsize']=(10,5)
plt.style.use('ggplot') 
pal = ["#9b59b6", "#e74c3c", "#34495e", "#2ecc71"]
plt.bar(time, withd, label = 'Withdrawal' )
plt.bar(time, inj , label = 'injection' )
plt.stackplot(time,volume_level[:,0], colors=pal, alpha=0.4, labels = 'Volume stored')
plt.legend(loc='upper right')
plt.title('Volume of natural gas stored')
plt.xlabel('Time (months)')
plt.ylabel('Volume [MWh]')
plt.show()
 


time = np.arange(0,14)  
plt.plot(time, facility.simulated_price_matrix_fwd[:,:])   
plt.title('Simulated prices')
plt.xlabel('Time (month)')
plt.ylabel('Price [EUR/MWh]')
plt.show()


  