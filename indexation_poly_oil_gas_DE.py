# -*- coding: utf-8 -*-
"""
Created on Mon May 14 09:18:18 2018

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
from time import time
import math
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
df_subset = df.dropna(how='any')
row = df_subset.shape[1]
col = df_subset.shape[0]
df_subset = df_subset[3:col]
russia_Index = df_subset[['Data Type','USD.20']].reset_index()[2:df_subset.shape[0]]
russia_Index.drop('index', axis=1, inplace=True)
russia_Index = russia_Index.reset_index()
russia_Index.drop('index', axis=1, inplace=True)
russia_Index.columns = ['date', 'index']
start_date_data = '1997-04-01 00:00:00'    
base =  datetime.strptime(start_date_data,"%Y-%m-%d %H:%M:%S")
nbr_months_per_data = 243
date_list = [base + relativedelta(months=x) for x in range(0, nbr_months_per_data)]
russia_Index['date'] = date_list

start_date_index = '2010-01-01 00:00:00'                                                                   # Select dates for Russia's index
end_date_index = '2015-12-01 00:00:00'  
mask = (russia_Index['date'] >= start_date_index) & (russia_Index['date'] <= end_date_index)
russia_Index_x = russia_Index.loc[mask].reset_index()  
russia_Index_x.drop('level_0', axis=1, inplace=True) 
y = russia_Index_x['index']





start_date_str = '2009-12-31 00:00:00'                                                       # Select dates for commodities prices
end_date_str = '2015-12-31 00:00:00'

start_date = datetime.strptime(start_date_str,"%Y-%m-%d %H:%M:%S") 
end_date = datetime.strptime(end_date_str,"%Y-%m-%d %H:%M:%S") 
df = pd.read_excel('spot_prices.xls')
df_oil = df[['date_oil', 'oil']]
df_oil.columns = ['date', 'oil']
df_oil['date'] = pd.to_datetime(df_oil['date'])  
mask = (df_oil['date'] >= start_date) & (df_oil['date'] <= end_date)
df_oil_x = df_oil.loc[mask].reset_index()  
df_oil_x.drop('index', axis=1, inplace=True) 
df_oil_x = df_oil_x.set_index('date')
df_oil_monthly = df_oil_x.resample("M", how='mean').reset_index().iloc[1:13,:]    


df_gas = df[['date_gas', 'gas']]
df_gas.columns = ['date', 'gas']
df_gas['date'] = pd.to_datetime(df_gas['date'])  
mask = (df_gas['date'] >= start_date) & (df_gas['date'] <= end_date)
df_gas_x = df_gas.loc[mask].reset_index()  
df_gas_x.drop('index', axis=1, inplace=True) 
df_gas_x = df_gas_x.set_index('date')
df_gas_monthly = df_gas_x.resample("M", how='mean').reset_index().iloc[1:13,:]   




def standardize(x):
    """Standardize the original data set."""
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


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
    # Define parameters to store w and loss
    #ws = [initial_w]
    #losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        #ws.append(w)
        #losses.append(loss)
        #perc_err = LA.norm(err)/LA.norm(y)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w={w}".format(
        #     bi=n_iter, ti=max_iters - 1, l=loss, w=w))       
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
    
    y_pred = ridgeregcv.predict(X_test)
    err = metrics.mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)  
    r2_train = r2_score(y_train, ridgeregcv.predict(X_train))
    score = ridgeregcv.score
    return ridgeregcv.coef_ , err, r2, r2_train, score
 

def linear_regression(X_train,y_train, X_test, y_test):
    
    regr = linear_model.LinearRegression() 
    regr.fit(X_train, y_train)  
    y_pred = regr.predict(X_test)    
    err =  mean_squared_error(y_test,y_pred)
    
    # Explained variance score: 1 is perfect prediction
    r2 = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, regr.predict(X_train))
    return regr.coef_ , err, r2, r2_train
    



def MA_func_vect_out(nbr_months_per_data, df1, start_date_str,end_date_str, lag, ma_period,reset_period,vect): #vect=np.empty(0) # pour reset_period=1 len(ma_vect)=11 au lieu de 12 je c pas pk
    start_date = datetime.strptime(start_date_str,"%Y-%m-%d %H:%M:%S") 
    end_date = datetime.strptime(end_date_str,"%Y-%m-%d %H:%M:%S") 
    start_date_x = start_date - relativedelta(months=int(round(lag))) - relativedelta(months=int(round(ma_period) ))            
    end_date =  end_date + relativedelta(months=int(round(lag))) + relativedelta(months=int(round(ma_period) ))+ relativedelta(months=1) 
    df1['date'] = pd.to_datetime(df1['date'])  
    mask = (df1['date'] >= start_date_x) & (df1['date'] <= end_date)
    df_x = df1.loc[mask].reset_index()  
    df_x.drop('index', axis=1, inplace=True)    
    df_x.iloc[:, [1]]= df_x.iloc[:, [1]].astype(float)   
    ma_monthly = pd.rolling_mean(df_x.set_index('date').resample('1BM'),window=int(round(ma_period))).dropna(how='any').reset_index().iloc[0:nbr_months_per_data, [1]].values
    ma_vect = [ ma_monthly[i] for i in range(0,nbr_months_per_data,int(round(reset_period))) ]
    nbr_reset_periods = int(math.ceil(nbr_months_per_data/int(round(reset_period))))  
    vect = np.empty(0)
    for i in range(0,nbr_reset_periods):
        vect = np.append(vect , ma_vect[i]*np.ones(int(round(reset_period))))    
    vect = vect[0:nbr_months_per_data]    
    return vect
     

def MA_plot(df1, start_date_str,end_date_str, lag, ma_period,reset_period):    
    nbr_months_per_data = 24
    vect = MA_func_vect_out(df1, start_date_str,end_date_str, lag, ma_period,reset_period,np.empty(0))
    time = np.arange(0,nbr_months_per_data)
    plt.plot(time, vect)   
    plt.title('Moving average: Lag = '+str(lag)+ ' MA period = '+ str(ma_period)+ ' Reset period = '+ str(reset_period)  )
    plt.xlabel('Time')
    plt.ylabel('Average level')
    plt.show()


 
def polynomial(X):    
    poly = PolynomialFeatures(degree=2) 
    return poly.fit_transform(X)       



def OLS_stat(X,y):
    """Summary statistics for OLS."""
    est = sm.OLS(y, X)
    est2 = est.fit()
    print(est2.summary())
 
 

class class_alternate(object):
    
    def __init__(self, df_oil, df_gas, y, start_date_str,end_date_str, nbr_months_per_data, nbr_iterations, max_lag, max_ma_period, max_reset_period ,init_coef):
            self.max_lag = max_lag
            self.max_ma_period = max_ma_period
            self.max_reset_period = max_reset_period
            
            self.nbr_iterations = nbr_iterations
 
            self.nbr_months_per_data = nbr_months_per_data
            self.bounds = [(0, self.max_lag),(0, self.max_lag), (1, self.max_ma_period),(1, self.max_ma_period),(1, self.max_reset_period),(1, self.max_reset_period)]
              
            self.df_oil = df_oil
            self.df_gas = df_gas
            self.y = y

            self.start_date_str = start_date_str                                             #start_date_str = '2016-01-31 00:00:00'
            self.start_date = datetime.strptime(self.start_date_str,"%Y-%m-%d %H:%M:%S") 
            self.end_date_str = end_date_str                                                 #end_date_str = '2017-01-31 00:00:00'
            self.end_date = datetime.strptime(self.end_date_str,"%Y-%m-%d %H:%M:%S") 
            
            self.init_coef = init_coef
            
    def MA_func_vect(self, lag_oil, lag_gas, ma_period_oil, ma_period_gas, reset_period_oil, reset_period_gas ,vect): #vect=np.empty(0) # pour reset_period=1 len(ma_vect)=11 au lieu de 12 je c pas pk
        
        start_date_oil = self.start_date - relativedelta(months=int(round(lag_oil))) - relativedelta(months=int(round(ma_period_oil) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_oil))) + relativedelta(months=int(round(ma_period_oil) ))+ relativedelta(months=1) #pour avoir 12 valeurs dans ma_vect
        self.df_oil['date'] = pd.to_datetime(self.df_oil['date'])  
        mask = (self.df_oil['date'] >= start_date_oil) & (self.df_oil['date'] <= end_date)
        df_xoil = self.df_oil.loc[mask].reset_index()  
        df_xoil.drop('index', axis=1, inplace=True)    
        df_xoil.iloc[:, [1]] = df_xoil.iloc[:, [1]].astype(float)                          
      
        start_date_gas = self.start_date - relativedelta(months=int(round(lag_gas))) - relativedelta(months=int(round(ma_period_gas) ))            
        end_date =   self.end_date + relativedelta(months=int(round(lag_gas))) + relativedelta(months=int(round(ma_period_gas) ))+ relativedelta(months=1) #pour avoir 12 valeurs dans ma_vect
        self.df_gas['date'] = pd.to_datetime(self.df_gas['date'])  
        mask = (self.df_gas['date'] >= start_date_gas) & (self.df_gas['date'] <= end_date)
        df_xgas = self.df_gas.loc[mask].reset_index()  
        df_xgas.drop('index', axis=1, inplace=True)    
        df_xgas.iloc[:, [1]] = df_xgas.iloc[:, [1]].astype(float)             
        
        ma_monthly_oil = pd.rolling_mean(df_xoil.set_index('date').resample('1BM'),window=int(round(ma_period_oil))).dropna(how='any').reset_index().iloc[0:self.nbr_months_per_data, [1]].values
        ma_monthly_gas = pd.rolling_mean(df_xgas.set_index('date').resample('1BM'),window=int(round(ma_period_gas))).dropna(how='any').reset_index().iloc[0:self.nbr_months_per_data, [1]].values
        
        
        ma_vect_oil = [ ma_monthly_oil[i] for i in range(0,self.nbr_months_per_data,int(round(reset_period_oil ))) ]
        ma_vect_gas = [ ma_monthly_gas[i] for i in range(0,self.nbr_months_per_data,int(round(reset_period_gas ))) ]
   
    
        nbr_reset_periods_oil = int(math.ceil(self.nbr_months_per_data/int(round(reset_period_oil))))  
        nbr_reset_periods_gas = int(math.ceil(self.nbr_months_per_data/int(round(reset_period_gas))))
        
        
        vect_oil = np.empty(0)
        for i in range(0,nbr_reset_periods_oil):
            vect_oil = np.append(vect_oil , ma_vect_oil[i]*np.ones(int(round(reset_period_oil))))      
        
        vect_gas = np.empty(0)
        for i in range(0,nbr_reset_periods_gas):
            vect_gas = np.append(vect_gas , ma_vect_gas[i]*np.ones(int(round(reset_period_gas))))      

        vect_oil = vect_oil[0:self.nbr_months_per_data]
        vect_gas = vect_gas[0:self.nbr_months_per_data]
        
        vect = np.c_[np.array(vect_oil) , np.array(vect_gas) ]
        
        return vect
    
        
    def func_lag_period(self, parameters, *data):
        
        lag_oil,  lag_gas, period_oil, period_gas, reset_oil,  reset_gas = parameters
        df_oil , df_gas , coef, values = data
        values = compute_mse( preprocessing.scale(self.y) ,  np.c_[np.ones(self.nbr_months_per_data), preprocessing.scale(polynomial(self.MA_func_vect(lag_oil, lag_gas, period_oil, period_gas, reset_oil,  reset_gas, np.empty(0))))[:,1:6].reshape(self.nbr_months_per_data,5)]  , coef) 
        return values



    def de_optimization(self, coef):
        """Differential evolution for the lag, ma_period and reset_period for oil, power, coal and gas."""        
        args = (self.df_oil, self.df_gas, coef, np.empty(1))
        result =differential_evolution(self.func_lag_period,  self.bounds, args=args)
        lag_oil, lag_gas, ma_period_oil, ma_period_gas,reset_period_oil, reset_period_gas = result.x                          
         
        return lag_oil, lag_gas, ma_period_oil, ma_period_gas,reset_period_oil, reset_period_gas
        
    
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
            lag_oil, lag_gas, ma_period_oil, ma_period_gas,reset_period_oil, reset_period_gas = self.de_optimization(coef)
            t11 = time()
            d1 = t11-t00
            print ("Duration of process 1 in Seconds %6.3f" %d1)      
            
            #process2: opt coeff given lag and period 
            t02 = time()
            X_df = self.MA_func_vect(lag_oil, lag_gas, ma_period_oil, ma_period_gas,reset_period_oil, reset_period_gas ,np.empty(0))     
            XX_stand = np.c_[np.ones(X_df.shape[0]), preprocessing.scale((polynomial(X_df))[:,1:6]).reshape(self.nbr_months_per_data,5)] 
            w_initial = gradient_w
            gradient_loss, gradient_w = gradient_descent(preprocessing.scale(self.y), XX_stand, w_initial, max_iters, gamma)            
            X_train, X_test, y_train, y_test = train_test_split(XX_stand, preprocessing.scale(self.y), random_state=1)
            
            res_ridge = ridge_regression(X_train, y_train, X_test, y_test)
             
            # update coef
            coef = res_ridge[0]
            y_pred_GD = np.dot(X_test,gradient_w)
            print('Coef GD:', gradient_w )
            print('Error GD:', metrics.mean_squared_error(y_test, y_pred_GD))
            print('R2_test GD', r2_score(y_test, y_pred_GD)  )
            print('R2_all_matrix GD', r2_score(preprocessing.scale(self.y), np.dot(XX_stand,gradient_w))  )
                        
            print('---------')            
            
            print('Coef RR:', res_ridge[0] )
            print('Error RR: ', res_ridge[1])
            print('R2 RR: ', res_ridge[2])
            print('R2_train RR: ', res_ridge[3])
            t12 = time()
            d2 = t12-t02
            print ("Duration of process 2 in Seconds %6.3f" % d2)        
            
            lag = np.c_[lag_oil, lag_gas]
            ma_period = np.c_[ma_period_oil, ma_period_gas]
            reset_period = np.c_[reset_period_oil, reset_period_gas]
        return coef , lag , ma_period, reset_period ,X_df, XX_stand, X_train, X_test, y_train, y_test
           
        


if __name__ == '__main__':
    
    #Step 1: optimizing lag, ma_period, reset_period and get the coefficients     
    
                                   #df_oil,  df_gas, y, start_date_str,end_date_str, nbr_months_per_data, nbr_iterations, max_lag, max_ma_period, max_reset_period ,init_coef
    optimization = class_alternate(df_oil ,df_gas,y ,'2014-01-31 00:00:00','2016-01-31 00:00:00', 24, 5, 12 , 12, 7, np.array([0,0,0,0,0,0]))
    t0 = time()
    coef , lag , ma_period, reset_period , X_df, XX_stand, X_train, X_test, y_train, y_test = optimization.alternate()    
    t1 = time()
    d = t1 - t0
    print ("Total duration in Seconds %6.3f" % d)               
    print('final coef: ', coef)
      
    
    OLS_stat(XX_stand,preprocessing.scale(y))
    
    
    
    # Testing 
    
    nbr_months_per_data = 12  
    start_date_index = '2016M1'                                                                
    end_date_index = '2017M1'
    start_indices = list(np.where(russia_Index["date"] == start_date_index)[0])[0]
    end_indices = list(np.where(russia_Index["date"] == end_date_index)[0])[0]
    yy_test = russia_Index.iloc[start_indices:end_indices ]
    y_test = yy_test['index']
    
    
    start_day = '2016-01-31 00:00:00'
    end_day = '2017-01-31 00:00:00'
    oil_test = MA_func_vect_out(nbr_months_per_data, df_oil, start_day , end_day , lag[0,0] , ma_period[0,0] , reset_period[0,0] , np.empty(0))
    gas_test = MA_func_vect_out(nbr_months_per_data, df_gas, start_day , end_day , lag[0,1],ma_period[0,1], reset_period[0,1] , np.empty(0))
    
    X_test = np.c_[oil_test,  gas_test]
    X_test_stand = np.c_[np.ones(X_test.shape[0]), preprocessing.scale((polynomial(X_test))[:,1:6]).reshape(nbr_months_per_data,5)]  
    
    OLS_stat(X_test_stand ,preprocessing.scale(y_test))
    
    error_test = compute_mse(preprocessing.scale(y_test), X_test_stand, coef)
    
    score(XX_stand,preprocessing.scale(y), X_test_stand, preprocessing.scale(y_test),coef )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    