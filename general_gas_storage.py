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



def payoff1(price ,dv, inj_cost , wth_cost, trans_cost):  
    h=  -price*dv*np.where(dv > 0, inj_cost + trans_cost, wth_cost + trans_cost)     
    return h


def payoff(S,dv,injection_cost,withdrawal_cost):   
    """specify the payoff."""
    return -withdrawal_cost*dv*S


def penalty(S,v):    
    """specify the penalty function (set to zero for simplicity)."""
    return 0
 
    

 


class gas_storage(object):
    """ 
    simulated_price_matrix : simulated gas prices
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
    

    """

    def __init__(self, simulated_price_matrix_fwd, T, steps, M, r, sigma_GBM, nbr_simulations, vMax, vMin, inj_rate, with_rate, injection_cost, withdrawal_cost):
     
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
              
    
    def max_inj_rates(self,vol):
        return min(self.vMax - vol , self.inj_rate )
        #return self.inj_rate

    def max_wit_rates(self,vol):
        return max(self.vMin - vol ,self.with_rate )
        #return self.with_rate
        
        
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
        simulated_price_matrix = np.zeros((self.steps + 2, self.nbr_simulations), dtype=np.float64)
        for b in range(0, self.nbr_simulations):
            simulated_price_matrix[:,b] = np.linspace(1,8,14)
        return simulated_price_matrix
      

    
    def contract_value(self):
        """Returns the value of the contract at time 0."""   
        value_matrix = np.zeros((self.simulated_price_matrix().shape[0],self.simulated_price_matrix().shape[1],self.M))  # time, path , volume level
        acc_cashflows = np.zeros_like(value_matrix)
        decision_rule = np.zeros_like(value_matrix)
        volume_level = np.zeros_like(self.simulated_price_matrix())
        
        decision_rule_avg = np.zeros((self.simulated_price_matrix().shape[0],self.M))
        volume_level_avg = np.zeros(self.simulated_price_matrix().shape[0])
        acc_cashflows_avg = np.zeros((self.simulated_price_matrix().shape[0],self.M))
        value_matrix[-1,: ,:] = penalty(self.simulated_price_matrix()[-1, :],volume_level[-1,:]) 
        acc_cashflows[-1,:,:] = penalty(self.simulated_price_matrix()[-1, :],volume_level[-1,:])
               
        for t in range(self.steps , -1 , -1):
        
            print ('-----------')
            print ('Time: %5.3f, Spot price: %5.1f ' % (t, self.simulated_price_matrix()[t, 1]))           
            for m in range(1,self.M):    
                
                volume_level[t+1,:] = (m-1)*self.alpha

                regression = np.polyfit(self.simulated_price_matrix()[t, :], acc_cashflows[t+1, :, m-1] * self.discount, 2)
                continuation_value = np.polyval(regression, self.simulated_price_matrix()[t, :])
                
                for b in range(self.nbr_simulations):
                     f = lambda x: -1*( payoff(self.simulated_price_matrix()[t, b],
                                               x ,self.injection_cost,self.withdrawal_cost ) + continuation_value[b]  )
    
                     cons = ({'type': 'ineq', 'fun': lambda x:  (volume_level[t+1,b] - x - self.vMin)            },   
                             {'type': 'ineq', 'fun': lambda x:  (self.vMax - volume_level[t+1,b] + x)            },
                             
                             {'type': 'ineq', 'fun': lambda x:  (self.max_inj_rates(volume_level[t+1,b]-x) - x)  },
                             {'type': 'ineq', 'fun': lambda x:  (x - self.max_wit_rates(volume_level[t+1,b]-x))  }, 
                     
                             
                             #{'type': 'ineq', 'fun': lambda x:  (self.inj_rate - x)                              },
                             #{'type': 'ineq', 'fun': lambda x:  (x - self.with_rate)                             },
                             
                             #{'type': 'ineq', 'fun': lambda x:  (volume_level[t+1,b] - x  - (np.abs(self.with_rate)*t)+ self.v_start  )   },                            
                             #{'type': 'ineq', 'fun': lambda x:  ((self.inj_rate*t + self.v_start) - volume_level[t+1,b] + x )      })   
        
                             {'type': 'ineq', 'fun': lambda x:  (self.max_inj_rates(volume_level[t+1,b]-x)*t - volume_level[t+1,b] + x ) })   
        
                     res = minimize(f, random.rand(1), constraints=cons)     
                     decision_rule[t,b,m-1] = res.x                  
                   
                acc_cashflows[t,:,m-1] = payoff(self.simulated_price_matrix()[t, :],
                             decision_rule[t,:,m-1] , self.injection_cost, self.withdrawal_cost) + acc_cashflows[t+1,:,m-1]*self.discount
                
                decision_rule_avg[t,m-1] = np.sum(decision_rule[t,:,m-1])/self.nbr_simulations
                volume_level_avg[t] = np.sum(volume_level[t,:])/self.nbr_simulations
                acc_cashflows_avg[t,m-1] = np.sum(acc_cashflows[t,:,m-1])/self.nbr_simulations
          
        contract_value = acc_cashflows[1,:,:] * self.discount             # at time 0
        
        return contract_value,acc_cashflows,decision_rule

 
    
    def price(self):
        return round((np.sum(self.contract_value()[0],axis=0) /  float(self.nbr_simulations))[0] , 2)
            
            



def volume_level_func1(decision_rule, steps, nbr_simulations, volume_end, alpha):
    decision_rule = np.around(decision_rule, decimals=-1)
    volume_level_stored = np.zeros((steps+2, nbr_simulations))
    volume_level_stored[steps+1,:] = volume_end
    for t  in range(steps,0,-1):   
            for b in range(nbr_simulations):      
                m_level = ( (volume_level_stored[t+1,b])/alpha).astype(int) + 1     
                if (np.abs(decision_rule[t,b,m_level]) >= alpha):                   
                       volume_level_stored[t,b] = volume_level_stored[t+1,b] - decision_rule[t,b,m_level]
                else:
                       volume_level_stored[t,b] = volume_level_stored[t+1,b]
    return volume_level_stored


def decision_volume_1(decision_rule, steps, nbr_simulations, volume_end, alpha, b_example):
    decision_rule = np.around(decision_rule, decimals=-1)
    volume_level_stored = volume_level_func1(decision_rule, steps, nbr_simulations, volume_end, alpha) 
    volume_level_stored_ex =volume_level_stored[:,b_example]  
    for t  in range(steps,-1,-1):                
            m_level = ( (volume_level_stored_ex[t+1])/alpha).astype(int) + 1    
            print('At time ', t , ', inject/withdraw: ',decision_rule[t,b_example,m_level] )           
       






def volume_level_func2(decision_rule, steps, nbr_simulations, volume_end, alpha):
    decision_rule = np.around(decision_rule, decimals=-1)
    volume_level_stored = np.zeros((steps+2, nbr_simulations))
    volume_level_stored[steps+1,:] = volume_end
    for t  in range(steps,0,-1):   
            for b in range(nbr_simulations):      
                m_level = ( (volume_level_stored[t+1,b])/alpha).astype(int)      -1               
                if (np.abs(decision_rule[t,b,m_level]) > alpha):                   
                       volume_level_stored[t,b] = volume_level_stored[t+1,b] - (np.sign(decision_rule[t,b,m_level]))*((abs(decision_rule[t,b,m_level])//alpha)+1)*alpha      
                elif ((np.abs(decision_rule[t,b,m_level]) % alpha) == 0):                   
                       volume_level_stored[t,b] = volume_level_stored[t+1,b] - decision_rule[t,b,m_level]
                else:
                       volume_level_stored[t,b] = volume_level_stored[t+1,b]
    return volume_level_stored
   

def decision_volume_2(decision_rule, steps, nbr_simulations, volume_end, alpha, b_example=0):
    decision_rule = np.around(decision_rule, decimals=-1)
    volume_level_stored = volume_level_func2(decision_rule, steps, nbr_simulations, volume_end, alpha) 
    volume_level_stored_ex =volume_level_stored[:,b_example]  
    for t  in range(steps,0,-1):   
            m_level = ( (volume_level_stored_ex[t+1])/alpha).astype(int) + 1    
            print('At time ', t , ', inject/withdraw: ',decision_rule[t,b_example,m_level] )           
    


def hist_with_inj(decision_rule, steps, nbr_simulations,volume_level_stored,  b_example):
    decision_rule = np.around(decision_rule, decimals=-1)
    volume_level_stored_ex = volume_level_stored[:,b_example]  
    inj = np.zeros(steps+2)
    withd = np.zeros(steps+2)
    for t  in range(steps, 0, -1):
        m_level = ( (volume_level_stored_ex[t+1])/alpha).astype(int) + 1    
        print('At time ', t , ', inject/withdraw: ',decision_rule[t,b_example,m_level] )           
        if decision_rule[t,b_example,m_level]>0 :
            inj[t] = decision_rule[t,b_example,m_level]
        else:
            withd[t] = np.abs(decision_rule[t, b_example, m_level])
        
    return withd, inj    
        


#### Step 4: Monte Carlo for gas storage
#facility =  gas_storage(simulated_price_matrix, 0 , 10000, 1, 12, 101, 0.06, 0.1, 50, 250000, 0 , 3500, -7500  ,0.1, 0.1)  
file = 'simulated_price_matrix_spot.xlsx'
simulated_price_matrix_spot = pd.read_excel(file)
simulated_price_matrix_spot = simulated_price_matrix_spot.values

file = 'simulated_price_matrix.xlsx'
simulated_price_matrix = pd.read_excel(file)
simulated_price_matrix = simulated_price_matrix.values
    
facility =  gas_storage(simulated_price_matrix, 1, 12, 101, 0.06, 0.1, 2, 250000, 0 , 25000, -75000  ,0.1, 0.1)  
contract_value, acc_cashflows, decision_rule = facility.contract_value() 



M = 101
nbr_simulations = 2
alpha = 2500
volume_end = 100000
steps = 12

volume_level1 = volume_level_func1(decision_rule, steps, nbr_simulations, volume_end, alpha)
decision_volume_1(decision_rule, steps, nbr_simulations, volume_end, alpha, 0)

volume_level2 = volume_level_func2(decision_rule, steps, nbr_simulations, volume_end, alpha)
decision_volume_2(decision_rule, steps, nbr_simulations, volume_end, alpha, 0)

withd, inj = hist_with_inj(decision_rule, steps, nbr_simulations, volume_level2 ,  0)



# library
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
time = np.arange(0,14)
plt.rcParams['figure.figsize']=(10,5)
plt.style.use('ggplot') 
 
pal = sns.color_palette("Set1")
#for i in range(M):
plt.bar(time, withd, label = 'Withdrawal' )
plt.bar(time, inj , label = 'injection' )
plt.stackplot(time,volume_level1[:,0], colors=pal, alpha=0.4, labels = 'Volume stored')
plt.legend(loc='upper right')
plt.title('Volume of natural gas stored 1')
plt.xlabel('Time (months)')
plt.ylabel('Volume [MWh]')
plt.show()
 



pal = ["#9b59b6", "#e74c3c", "#34495e", "#2ecc71"]
#for i in range(M):
plt.bar(time, withd, label = 'Withdrawal' )
plt.bar(time, inj , label = 'injection' )
plt.stackplot(time, volume_level2[:,0], colors=pal, alpha=0.4, labels = 'Volume stored')
plt.legend(loc='upper left')
plt.title('Volume of natural gas stored 2')
plt.xlabel('Time (months)')
plt.ylabel('Volume [MWh]')
plt.show()



    
from time import time
t0 = time()
print ('Price: ', facility.price() )   
t1 = time();
d1 = t1 - t0
print ("Duration in Seconds %6.3f" % d1)       


time = np.arange(0,14)  
plt.plot(time, facility.simulated_price_matrix()[:,:])   
#plt.plot(time, facility.simulated_price_matrix[:,:])   
plt.title('simulated_price_matrix')
plt.xlabel('Time')
plt.ylabel('Price [EUR/MWh]')
plt.show()


