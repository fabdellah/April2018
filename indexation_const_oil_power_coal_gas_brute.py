# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 16:54:03 2018

@author: fabdellah
"""

#MemoryError

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta, date, datetime
from dateutil.relativedelta import relativedelta
from scipy import optimize
from numpy import linalg as LA
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import RidgeCV
import warnings
warnings.filterwarnings("ignore")
from time import time


import warnings
warnings.filterwarnings("ignore")

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats


# Import data
file = 'External_Data.xls'
df = pd.read_excel(file)
#print(df.columns)
df_subset = df.dropna(how='any')
row = df_subset.shape[1]
col = df_subset.shape[0]
df_subset = df_subset[3:col]
russia_Index = df_subset[['Data Type','USD.20']].reset_index()[2:df_subset.shape[0]]
russia_Index.drop('index', axis=1, inplace=True)
russia_Index = russia_Index.reset_index()

df = pd.read_excel('spot_prices.xls')
df_oil = df[['date_oil', 'oil']]
df_oil.columns = ['date', 'oil']

df_power = df[['date_power', 'power']]
df_power.columns = ['date', 'power']

df_coal = df[['date_coal', 'coal']]
df_coal.columns = ['date', 'coal']

df_gas = df[['date_gas', 'gas']]
df_gas.columns = ['date', 'gas']


yy = russia_Index[225:237]           #monthly from 2016 to 2017
yy = yy.reset_index()  
yy.drop('index', axis=1, inplace=True)
yy.drop('level_0', axis=1, inplace=True)
y = yy['USD.20']




def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)



def compute_mse_func(y, x1,x2,x3,x4, coef1,coef2,coef3,coef4):
    """compute the loss by mse."""
    e = y - (x1*coef1 + x2*coef2 + x3*coef3 + x4*coef4)
    mse = e.dot(e) / (2 * len(y))
    return mse


def compute_mse(y, x, coef):
    """compute the loss by mse."""
    e = y - np.dot(x,coef)
    mse = e.dot(e) / (2 * len(y))
    return mse


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - np.dot(tx,w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def predict(tx,coef):
    #tx = df[['oil','power','coal','gas']].values
    return np.dot(tx,coef)
    

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for linear regression."""
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
    return loss, w

def score(X_train,y_train, X_test, y_test,coef):
    y_pred_train = np.dot(X_train,coef)     
    r2 = r2_score(y_test, np.dot(X_test,coef))  
    r2_train = r2_score(y_train, y_pred_train)
    return r2,r2_train


def ridge_regression(X_train,y_train, X_test, y_test):    
    """Ridge regression algorithm."""
    # select the best alpha with RidgeCV (cross-validation)
    # alpha=0 is equivalent to linear regression
    alpha_range = 10.**np.arange(-2, 3)
    ridgeregcv = RidgeCV(alphas=alpha_range, normalize=False, scoring='mean_squared_error') 
    ridgeregcv.fit(X_train, y_train)
    #print('best alpha=',ridgeregcv.alpha_)
    #print('ridgeregcv.coef: ',ridgeregcv.coef_)
    # predict method uses the best alpha value
    y_pred = ridgeregcv.predict(X_test)
    err = metrics.mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)  
    r2_train = r2_score(y_train, ridgeregcv.predict(X_train))
    score = ridgeregcv.score
    return ridgeregcv.coef_ , err, r2, r2_train, score
 

    
def MA_func_vect_out(df1, start_date_str,end_date_str, lag, ma_period,reset_period,vect): #vect=np.empty(0) # pour reset_period=1 len(ma_vect)=11 au lieu de 12 je c pas pk
    nbr_months_per_year = 12
    start_date = datetime.strptime(start_date_str,"%Y-%m-%d %H:%M:%S") 
    end_date = datetime.strptime(end_date_str,"%Y-%m-%d %H:%M:%S") 
    start_date_x = start_date - relativedelta(months=int(round(lag))) - relativedelta(months=int(round(ma_period) ))            
    end_date =  end_date + relativedelta(months=int(round(lag))) + relativedelta(months=int(round(ma_period) ))+ relativedelta(months=1) 
    df1['date'] = pd.to_datetime(df1['date'])  
    mask = (df1['date'] >= start_date_x) & (df1['date'] <= end_date)
    df_x = df1.loc[mask].reset_index()  
    df_x.drop('index', axis=1, inplace=True)    
    df_x.iloc[:, [1]]= df_x.iloc[:, [1]].astype(float)   
    ma_monthly = pd.rolling_mean(df_x.set_index('date').resample('1BM'),window=int(round(ma_period))).dropna(how='any').reset_index().iloc[0:12, [1]].values
    ma_vect = [ ma_monthly[i] for i in range(0,nbr_months_per_year,int(round(reset_period))) ]
    nbr_months_per_year = 12
    nbr_reset_periods = int(nbr_months_per_year/reset_period)
    vect = np.empty(0)
    for i in range(0,nbr_reset_periods):
        #print('ma_vect[i]',i, ma_vect[i])           
        vect = np.append(vect , ma_vect[i]*np.ones(int(round(reset_period))))      
    return vect
     

def MA_plot(df1, start_date_str,end_date_str, lag, ma_period,reset_period):    
    nbr_months_per_year = 12
    vect = MA_func_vect_out(df1, start_date_str,end_date_str, lag, ma_period,reset_period,np.empty(0))
    time = np.arange(0,nbr_months_per_year)
    plt.plot(time, vect)   
    plt.title('Moving average: Lag = '+str(lag)+ ' MA period = '+ str(ma_period)+ ' Reset period = '+ str(reset_period)  )
    plt.xlabel('Time')
    plt.ylabel('Average level')
    plt.show()
 
 

class class_alternate(object):
    
    def __init__(self, df_oil, df_power, df_coal, df_gas, y, start_date_str,end_date_str, nbr_months_per_year, nbr_iterations, max_lag, max_ma_period, max_reset_period ,init_coef):
            self.max_lag = max_lag
            self.max_ma_period = max_ma_period
            self.max_reset_period = max_reset_period
            
            self.nbr_iterations = nbr_iterations
 
            self.nbr_months_per_year = nbr_months_per_year
            self.bounds = [(1, self.max_lag), (1, self.max_ma_period), (2, self.max_reset_period)]
              
            self.df_oil = df_oil
            self.df_power = df_power
            self.df_coal = df_coal
            self.df_gas = df_gas
            self.y = y

            self.start_date_str = start_date_str                                             #start_date_str = '2016-01-31 00:00:00'
            self.start_date = datetime.strptime(self.start_date_str,"%Y-%m-%d %H:%M:%S") 
            self.end_date_str = end_date_str                                                 #end_date_str = '2017-01-31 00:00:00'
            self.end_date = datetime.strptime(self.end_date_str,"%Y-%m-%d %H:%M:%S") 
            
            self.rranges = (slice(1, self.max_lag, 1),slice(1, self.max_lag, 1),slice(1, self.max_lag, 1),slice(1, self.max_lag, 1), slice(1, self.max_ma_period, 1),slice(1, self.max_ma_period, 1),slice(1, self.max_ma_period, 1),slice(1, self.max_ma_period, 1), slice(3, max_reset_period, 3),slice(3, max_reset_period, 3),slice(3, max_reset_period, 3),slice(3, max_reset_period, 3))
            self.init_coef = np.array([init_coef,init_coef,init_coef,init_coef])
            
    def MA_func_vect(self, lag_oil, lag_power, lag_coal, lag_gas, ma_period_oil, ma_period_power, ma_period_coal, ma_period_gas, reset_period_oil, reset_period_power, reset_period_coal, reset_period_gas ,vect): #vect=np.empty(0) # pour reset_period=1 len(ma_vect)=11 au lieu de 12 je c pas pk
        
        start_date_oil = self.start_date - relativedelta(months=int(round(lag_oil))) - relativedelta(months=int(round(ma_period_oil) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_oil))) + relativedelta(months=int(round(ma_period_oil) ))+ relativedelta(months=1) #pour avoir 12 valeurs dans ma_vect
        self.df_oil['date'] = pd.to_datetime(self.df_oil['date'])  
        mask = (self.df_oil['date'] >= start_date_oil) & (self.df_oil['date'] <= end_date)
        df_xoil = self.df_oil.loc[mask].reset_index()  
        df_xoil.drop('index', axis=1, inplace=True)    
        df_xoil.iloc[:, [1]] = df_xoil.iloc[:, [1]].astype(float)          
                
        start_date_power = self.start_date - relativedelta(months=int(round(lag_power))) - relativedelta(months=int(round(ma_period_power) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_power))) + relativedelta(months=int(round(ma_period_power) ))+ relativedelta(months=1) #pour avoir 12 valeurs dans ma_vect
        self.df_power['date'] = pd.to_datetime(self.df_power['date'])  
        mask = (self.df_power['date'] >= start_date_power) & (self.df_power['date'] <= end_date)
        df_xpower = self.df_power.loc[mask].reset_index()  
        df_xpower.drop('index', axis=1, inplace=True)    
        df_xpower.iloc[:, [1]] = df_xpower.iloc[:, [1]].astype(float)            
    
        start_date_coal = self.start_date - relativedelta(months=int(round(lag_coal))) - relativedelta(months=int(round(ma_period_coal) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_coal))) + relativedelta(months=int(round(ma_period_coal) ))+ relativedelta(months=1) #pour avoir 12 valeurs dans ma_vect
        self.df_coal['date'] = pd.to_datetime(self.df_coal['date'])  
        mask = (self.df_coal['date'] >= start_date_coal) & (self.df_oil['date'] <= end_date)
        df_xcoal = self.df_coal.loc[mask].reset_index()  
        df_xcoal.drop('index', axis=1, inplace=True)    
        df_xcoal.iloc[:, [1]] = df_xcoal.iloc[:, [1]].astype(float)          
      
        start_date_gas = self.start_date - relativedelta(months=int(round(lag_gas))) - relativedelta(months=int(round(ma_period_gas) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_gas))) + relativedelta(months=int(round(ma_period_gas) ))+ relativedelta(months=1) #pour avoir 12 valeurs dans ma_vect
        self.df_gas['date'] = pd.to_datetime(self.df_oil['date'])  
        mask = (self.df_gas['date'] >= start_date_gas) & (self.df_gas['date'] <= end_date)
        df_xgas = self.df_gas.loc[mask].reset_index()  
        df_xgas.drop('index', axis=1, inplace=True)    
        df_xgas.iloc[:, [1]] = df_xgas.iloc[:, [1]].astype(float)             
        
        ma_monthly_oil = pd.rolling_mean(df_xoil.set_index('date').resample('1BM'),window=int(round(ma_period_oil))).dropna(how='any').reset_index().iloc[0:12, [1]].values
        ma_monthly_power = pd.rolling_mean(df_xpower.set_index('date').resample('1BM'),window=int(round(ma_period_power))).dropna(how='any').reset_index().iloc[0:12, [1]].values
        ma_monthly_coal = pd.rolling_mean(df_xcoal.set_index('date').resample('1BM'),window=int(round(ma_period_coal))).dropna(how='any').reset_index().iloc[0:12, [1]].values
        ma_monthly_gas = pd.rolling_mean(df_xgas.set_index('date').resample('1BM'),window=int(round(ma_period_gas))).dropna(how='any').reset_index().iloc[0:12, [1]].values
        
        ma_monthly = np.append(ma_monthly_oil , ma_monthly_power , ma_monthly_coal , ma_monthly_gas)
        print('ma_monthly : type: ', type(ma_monthly))        
        print('-----------------------------------------')
        print('lag oil = ', lag_oil, ' ma_period_oil =  ', ma_period_oil, ' reset period_oil  = ' , round(reset_period_oil))
        ma_vect_oil = [ ma_monthly_oil[i] for i in range(0,self.nbr_months_per_year,int(round(reset_period_oil ))) ]
        ma_vect_power = [ ma_monthly_power[i] for i in range(0,self.nbr_months_per_year,int(round(reset_period_power ))) ]
        ma_vect_coal = [ ma_monthly_coal[i] for i in range(0,self.nbr_months_per_year,int(round(reset_period_coal ))) ]
        ma_vect_gas = [ ma_monthly_gas[i] for i in range(0,self.nbr_months_per_year,int(round(reset_period_gas ))) ]
   
        nbr_reset_periods_oil = int(self.nbr_months_per_year/int(round(reset_period_oil )))
        nbr_reset_periods_power = int(self.nbr_months_per_year/int(round(reset_period_power )))
        nbr_reset_periods_coal = int(self.nbr_months_per_year/int(round(reset_period_coal )))
        nbr_reset_periods_gas = int(self.nbr_months_per_year/int(round(reset_period_gas )))
        
        vect_oil = np.empty(0)
        for i in range(0,nbr_reset_periods_oil):
            vect_oil = np.append(vect_oil , ma_vect_oil[i]*np.ones(int(round(reset_period_oil))))      
        print('vect_oil: ',vect_oil)
        
        vect_power = np.empty(0)
        for i in range(0,nbr_reset_periods_power):
            vect_power = np.append(vect_power , ma_vect_power[i]*np.ones(int(round(reset_period_power))))      
        print('vect_oil: ',vect_power)
        
        vect_coal = np.empty(0)
        for i in range(0,nbr_reset_periods_coal):
            vect_coal = np.append(vect_coal , ma_vect_coal[i]*np.ones(int(round(reset_period_coal))))      
        print('vect_oil: ',vect_coal)
        
        vect_gas = np.empty(0)
        for i in range(0,nbr_reset_periods_gas):
            vect_gas = np.append(vect_gas , ma_vect_gas[i]*np.ones(int(round(reset_period_gas))))      
        print('vect_oil: ',vect_gas)
              
        vect = np.append(vect_oil,vect_power,vect_coal,vect_gas)
        
        return vect
    
        
    def func_lag_period(self, parameters, *data):
        
        lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas = parameters
        df_oil , df_power , df_coal , df_gas , y, coef, values = data
        print('ici func_lag_period')
        values = compute_mse(self.y , self.MA_func_vect(lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas, np.empty(0)), coef) 
        return values


    def brute_optimization(self,  coef):
                
        args = (self.df_oil,self.df_power,self.df_coal,self.df_gas, self.y, coef, np.empty(1))
        
        lag_oil, lag_power, lag_coal , lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas = optimize.brute(self.func_lag_period, self.rranges, args=args, full_output=True,
                              finish=optimize.fmin)[0]
        return lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas
        
    
    def alternate(self):
        max_iters = 100
        gamma = 0.1
        coef =  self.init_coef
        gradient_w = self.init_coef    
         
        for itera in range(self.nbr_iterations):
            
            print('///////////////////////////////////////')
            print('Iteration: ', itera )
            
            #process 1: opt lag and opt period given coef
           
            t00 = time()
            lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas = self.brute_optimization(coef)
            t11 = time()
            d1 = t11-t00
            lag = np.array([lag_oil, lag_power, lag_coal, lag_gas])
            ma_period = np.array([period_oil, period_power, period_coal, period_gas])
            reset_period = np.array([reset_oil, reset_power, reset_coal, reset_gas])
            print ("Duration of process 1 in Seconds %6.3f" %d1)      
            
            #process2: optimal coefficient for a given lag, ma_period and reset period
            t02 = time()
            X_df = self.MA_func_vect(lag_oil, lag_power, lag_coal, lag_gas, period_oil, period_power, period_coal, period_gas, reset_oil, reset_power, reset_coal, reset_gas, np.empty(0))     
            XX_stand = np.c_[np.ones(X_df.shape[0]), preprocessing.scale(X_df).reshape(self.nbr_months_per_year,4)] 
            w_initial = gradient_w
            #gradient_loss, gradient_w = gradient_descent(preprocessing.scale(self.y), XX_stand, w_initial, max_iters, gamma)            
            X_train, X_test, y_train, y_test = train_test_split(XX_stand, preprocessing.scale(self.y), random_state=1)
            
            gradient_loss, gradient_w = gradient_descent(preprocessing.scale(y_train), X_train, w_initial, max_iters, gamma)  
            res_ridge = ridge_regression(X_train, y_train, X_test, y_test)
             
            # update coef
            coef = res_ridge[0]
            y_pred_GD = np.dot(X_test,gradient_w)
            print('--------- Gradient descent ---------')
            print('Coef GD:', gradient_w )
            print('Error GD:', metrics.mean_squared_error(y_test, y_pred_GD))
            print('R2_train GD', r2_score(preprocessing.scale(y_train), np.dot(X_train,gradient_w))  )
            print('R2_test GD', r2_score(y_test, y_pred_GD)  )
                                
            print('--------- Ridge regression ---------')
            
            print('Coef RR:', res_ridge[0] )
            print('Error RR: ', res_ridge[1])
            print('R2_train RR: ', res_ridge[3])
            print('R2_test RR: ', res_ridge[2])
            
            t12 = time()
            d2 = t12-t02
            print ("Duration of process 2 in Seconds %6.3f" % d2)        
                
        return coef , lag , ma_period, reset_period ,X_df, XX_stand, X_train, X_test, y_train, y_test


if __name__ == '__main__':
    
    #Step 1: optimizing lag, ma_period, reset_period and get the coefficients     
    #df_oil, start_date_str,end_date_str, nbr_months_per_year, nbr_iterations, max_lag, max_ma_period, max_reset_period ,init_coef
    optimization = class_alternate(df_oil, df_power ,df_coal ,df_gas ,y, '2016-01-31 00:00:00','2017-01-31 00:00:00', 12, 5, 12 , 12, 7, 0)
    t0 = time()
    coef , lag , ma_period, reset_period , X_train, X_test, y_train, y_test   = optimization.alternate()    
    t1 = time()
    d = t1 - t0
    print ("Total duration in Seconds %6.3f" % d)               
    print('final coef: ', coef)
    
    
    MA_plot(df_oil, '2016-01-31 00:00:00','2017-01-31 00:00:00', lag , ma_period, reset_period)
    
    lag=1
    ma_period = 2
    reset_period = 2
    
    v = optimization.MA_func_vect(lag, ma_period,reset_period, np.empty(0))
    
    
    