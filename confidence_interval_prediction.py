# -*- coding: utf-8 -*-
"""
Created on Tue May 15 12:31:45 2018

@author: fabdellah
"""

# Source: https://github.com/urgedata/pythondata/blob/master/fbprophet/fbprophet_part_one.ipynb

#### confidence interval prophet



import matplotlib.pyplot as plt
import datetime
import numpy as np
import pandas as pd
from datetime import timedelta, date, datetime
from dateutil.relativedelta import relativedelta
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 

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
russia_Index.columns = ['date', 'rus_index']
start_date_data = '1997-04-01'    
base =  datetime.strptime(start_date_data,"%Y-%m-%d")
nbr_months_per_data = 243
date_list = [base + relativedelta(months=x) for x in range(0, nbr_months_per_data)]
russia_Index['date'] = date_list

start_date_index = '1997-04-01'                                                                    # Select dates for Russia's index
end_date_index =  '2015-12-01'
mask = (russia_Index['date'] >= start_date_index) & (russia_Index['date'] <= end_date_index)
russia_Index_x = russia_Index.loc[mask].reset_index()  
russia_Index_x.drop('index', axis=1, inplace=True) 



start_date_index =  '1997-04-01'                                                              # Select dates for Russia's index
end_date_index =  '2016-12-01'                 
mask = (russia_Index['date'] >= start_date_index) & (russia_Index['date'] <= end_date_index)
russia_Index_true = russia_Index.loc[mask].reset_index()  
russia_Index_true.drop('index', axis=1, inplace=True) 
russia_Index_true.set_index('date', inplace=True)


#Prophet
 
plt.rcParams['figure.figsize']=(10,5)
plt.style.use('ggplot')



df = russia_Index_x.reset_index()
df.drop('index', axis=1, inplace=True) 


df=df.rename(columns={'date':'ds', 'rus_index':'y'})
df['rus_index'] = df['y']
df.set_index('ds').y.plot()


df['y'] = np.log(np.float64((df['y']).values))  
df.set_index('ds').y.plot()

model = Prophet()
model.fit(df)


future = model.make_future_dataframe(periods=12, freq = 'MS')


forecast = model.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

model.plot(forecast)


df.set_index('ds', inplace=True)
forecast.set_index('ds', inplace=True)


viz_df = df.join(forecast[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')

viz_df['yhat_rescaled'] = np.exp(viz_df['yhat'])

viz_df[['rus_index', 'yhat_rescaled']].plot()

russia_Index_x.index = pd.to_datetime(russia_Index_x.index) #make sure our index as a datetime object
connect_date = df.index[-2] #select the 2nd to last date


mask = (forecast.index > connect_date)
predict_df = forecast.loc[mask]
actual_df = russia_Index_true.loc[mask]

viz_df = df.join(predict_df[['yhat', 'yhat_lower','yhat_upper']], how = 'outer')
viz_df['yhat_scaled'] = np.exp(viz_df['yhat'])
viz_df['actual_prices'] = pd.Series(russia_Index_true['rus_index'], index=viz_df.index)   


fig, ax1 = plt.subplots()
ax1.plot(viz_df.actual_prices, color='red')
ax1.plot(viz_df.yhat_scaled, color='black', linestyle=':')
ax1.fill_between(viz_df.index, np.exp(viz_df['yhat_upper']), np.exp(viz_df['yhat_lower']), alpha=0.5, color='darkgray')
ax1.set_title('Russian gas prices: Actual vs forecast')
ax1.set_ylabel('US Dollar')
ax1.set_xlabel('Date')
L=ax1.legend() #get the legend
L.get_texts()[0].set_text('Actual prices') #change the legend text for 1st plot
L.get_texts()[1].set_text('Forecast prices') #change the legend text for 2nd plot



#Metrics   #source et interpretation: https://pythondata.com/forecasting-time-series-data-prophet-part-4/

metric_df = np.exp(forecast[['yhat']]).join(np.exp(df.y)).reset_index()
metric_df.dropna(inplace=True)
r2_score(metric_df.y,  metric_df.yhat )
mean_squared_error(metric_df.y, metric_df.yhat)
mean_absolute_error(metric_df.y, metric_df.yhat)












####################################################################################################################################


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
start_date_data = '1997-04-01'    
base =  datetime.strptime(start_date_data,"%Y-%m-%d")
nbr_months_per_data = 243
date_list = [base + relativedelta(months=x) for x in range(0, nbr_months_per_data)]
russia_Index['date'] = date_list

start_date_index = '1997-04-01'                                                                    # Select dates for Russia's index
end_date_index = '2015-12-01'
mask = (russia_Index['date'] >= start_date_index) & (russia_Index['date'] <= end_date_index)
russia_Index_x = russia_Index.loc[mask].reset_index()  
russia_Index_x.drop('level_0', axis=1, inplace=True) 

russia_Index_x.columns = ['ds', 'y']



russia_Index_x['y_orig'] = russia_Index_x['y'] # to save a copy of the original data..you'll see why shortly. 

russia_Index_x['y'] =  np.log(np.float64((russia_Index_x['y']).values))  # log-transform y


model = Prophet() #instantiate Prophet
model.fit(russia_Index_x); #fit the model with your dataframe

future_data = model.make_future_dataframe(periods=6, freq = 'm')
forecast_data = model.predict(future_data)
forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

model.plot(forecast_data)

model.plot_components(forecast_data)



#forecast with original scale

forecast_data_orig = forecast_data # make sure we save the original forecast data
forecast_data_orig['yhat'] = np.exp(forecast_data_orig['yhat'])
forecast_data_orig['yhat_lower'] = np.exp(forecast_data_orig['yhat_lower'])
forecast_data_orig['yhat_upper'] = np.exp(forecast_data_orig['yhat_upper'])



russia_Index_x['y_log'] = russia_Index_x['y'] #copy the log-transformed data to another column
russia_Index_x['y'] = russia_Index_x['y_orig'] #copy the original data to 'y'

model.plot(forecast_data_orig)



