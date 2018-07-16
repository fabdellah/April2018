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




def volume_level_func(decision_rule, steps, nbr_simulations, volume_start, alpha):
    """Returns the optimal volume stored at each time given a volume at the starting time"""
    volume_level_stored = np.zeros((steps+1, nbr_simulations))  
    volume_level_stored[0,:] = volume_start
    for b in range(nbr_simulations):   
        for t  in range(steps): 
            m_level = ( (volume_level_stored[t,b])/alpha).astype(int) 
            volume_level_stored[t+1,b] = volume_level_stored[t,b] + decision_rule[t,b,m_level]
    volume_level_avg =    np.zeros(steps+1)     
    volume_level_avg = (np.sum(volume_level_stored,axis =1)/nbr_simulations).astype(int)
    return volume_level_stored, volume_level_avg


def decision_volume(decision_rule, steps, nbr_simulations, volume_start, alpha):
    """Returns the optimal behaviour (inject or withdraw) at each time given a volume at the final time"""
    inj = np.zeros((steps+1, nbr_simulations))  
    withd = np.zeros((steps+1, nbr_simulations))  
    volume_level_stored, volume_level_avg = volume_level_func(decision_rule, steps, nbr_simulations, volume_start, alpha) 
    
    for b in range(nbr_simulations):   
        for t in range(steps):               
            m_level = ( (volume_level_stored[t,b])/alpha).astype(int)     
            if decision_rule[t,b,m_level]>0 :
                inj[t,b] = decision_rule[t,b,m_level]
            else:
                withd[t,b] = np.abs(decision_rule[t, b, m_level])
    inj_avg = np.zeros(steps+1)     
    inj_avg = (np.sum(inj,axis =1)/nbr_simulations).astype(int)
    withd_avg = np.zeros(steps+1)     
    withd_avg = (np.sum(withd,axis =1)/nbr_simulations).astype(int)
    return withd_avg, inj_avg


    

 


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
    
    """

    def payoff(self,S, dv, injection_cost, withdrawal_cost):   
        """specify the payoff."""
        #return S*dv*np.where(dv > 0, -injection_cost, -withdrawal_cost )    
        return -S*dv
    
    def penalty(self,v,v_target):    
        """specify the penalty function."""       
        penalty = -np.abs(v-v_target)*100
        return penalty
    

    def __init__(self, simulated_price_matrix_fwd, T, steps, M, r, sigma_GBM, nbr_simulations, vMax, vMin, inj_rate, with_rate, injection_cost, withdrawal_cost, v_end):
     
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
              
            self.v_end = v_end
            
     
        
    def volume_limit_up(self):
        limit_up = np.zeros(self.steps+1)
        for t in range(self.steps+1):
            limit_up[t] = np.minimum(self.v_start + t*self.alpha , self.v_end + (self.steps-t)*self.alpha)
        return limit_up
    

    def volume_limit_down(self):
        limit_down = np.zeros(self.steps+1)
        for t in range(self.steps+1):
            limit_down[t] = np.maximum(self.v_start - t*self.alpha , self.v_end - (self.steps-t)*self.alpha)
        return limit_down
    
    
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
#            simulated_price_matrix[:,b] = np.linspace(1, 14, self.steps + 2)      #contango
            simulated_price_matrix[:,b] = np.linspace(14, 1, self.steps + 2)      #backwardation
#            simulated_price_matrix[:,b] = np.array([7,6,5,4,3,2,1,2,3,4,5,6,7,8])

        return simulated_price_matrix
      
        
    
    def f(self,x,t,b,m):
        return ( self.payoff(self.simulated_price_matrix()[t, b], x ,self.injection_cost,self.withdrawal_cost ) + self.continuation_value[t,b, (m-1) + int(x/self.alpha) ]  )
  
    
    
    def contract_value(self):
        """Returns the value of the contract at time 0, the accumulated cashflows, the decision rule and the final price of the storage."""   
        
        levels_volume = np.linspace(0,self.alpha*100,101)
        acc_cashflows = np.zeros((self.steps+1, self.nbr_simulations, self.M))  # time, path , volume level
        acc_cashflows[-1,:, :] = self.penalty(levels_volume, self.v_end)
        
        decision_rule = np.empty((self.steps+1, self.nbr_simulations, self.M))  # time, path , volume level
        decision_rule[:] = np.nan
        
        self.continuation_value = np.zeros((self.steps, self.nbr_simulations, self.M)) 
        
        
        
        for t in range(self.steps-1 , -1  , -1):
        
            print ('-----------')
            print ('Time: %5.3f, Spot price: %5.1f ' % (t, self.simulated_price_matrix()[t, 1]))           
            for m in range(1,self.M):                      
                                   
                    regression = np.polyfit(self.simulated_price_matrix()[t, :], acc_cashflows[t+1, :, m-1] * self.discount, 2)
                    self.continuation_value[t,:,m-1] = np.polyval(regression, self.simulated_price_matrix()[t, :])
            
            for b in range(self.nbr_simulations):
                for m in range(1,self.M): 
                                      
                     possible_values=[-self.alpha,0,self.alpha]
                     #possible_values=[0,self.alpha]
                     out= np.array([self.f(x,t,b,m) for x in possible_values])
                     
                     arg_maximum= np.argmax(out)
                     
                     dv_opt= possible_values[arg_maximum]
                                              
                     decision_rule[t,b,m-1] = dv_opt  
                                         
                     acc_cashflows[t,b,m-1] = self.payoff(self.simulated_price_matrix()[t, b],
                                     decision_rule[t,b,m-1] , self.injection_cost, self.withdrawal_cost) + acc_cashflows[t+1 , b,  (m-1) + int((decision_rule[t,b , m-1])/self.alpha)  ]*self.discount
                        
                  
        contract_value = acc_cashflows[1,:,:] * self.discount             # at time 0
        
        price = round((np.sum(contract_value,axis=0) /  float(self.nbr_simulations))[0] , 2)
        
        return contract_value, acc_cashflows, decision_rule, price






file = 'simulated_price_matrix.xlsx'            #not used for now
simulated_price_matrix = pd.read_excel(file)
simulated_price_matrix = simulated_price_matrix.values
    


facility =  gas_storage(simulated_price_matrix, 1, 12, 101, 0.035, 0.1, 2, 250000, 0 , 25000, -25000  , 0.1, 0.1 , 100000)  
contract_value, acc_cashflows, decision_rule, price = facility.contract_value() 



M = 101
nbr_simulations = 2
alpha = 2500
volume_start = 100000
steps = 12

volume_level,  volume_level_avg = volume_level_func(decision_rule, steps, nbr_simulations, volume_start, alpha)
withd, inj = decision_volume(decision_rule, steps, nbr_simulations, volume_start, alpha)





time = np.arange(0,14)  
plt.plot(time, facility.simulated_price_matrix()  ) 
plt.title('Forward curve')
plt.xlabel('Time (month)')
plt.ylabel('Price [EUR/MWh]')
plt.show()


# Plot the optimal volume stored (example)
ymini = 80000
ymaxi = 120000
time = np.arange(0,13)
plt.rcParams['figure.figsize']=(10,5)
plt.style.use('ggplot') 
pal = ["#9b59b6", "#e74c3c", "#34495e", "#2ecc71"]
plt.bar(time, withd+ymini, label = 'Withdrawal' )
plt.bar(time, inj+ymini , label = 'injection' )
plt.stackplot(time,volume_level_avg, colors=pal, alpha=0.4, labels = 'Volume stored')
plt.ylim(ymin = ymini, ymax = ymaxi) 
plt.legend(loc='upper right')
plt.title('Volume of natural gas imported')
plt.xlabel('Time (months)')
plt.ylabel('Volume [MWh]')
plt.show()
 


time = np.arange(0,14)  
plt.plot(time, facility.simulated_price_matrix_fwd)   
plt.title('Simulated prices')
plt.xlabel('Time (month)')
plt.ylabel('Price [EUR/MWh]')
plt.show()