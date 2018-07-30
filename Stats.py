# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:25:21 2018

@author: fabdellah
"""

import pandas as pd

# Import data
# data from: http://ec.europa.eu/eurostat/cache/infographs/energy/bloc-2c.html
file = 'stats.xlsx'
df = pd.read_excel(file)
print(df.columns)


from bokeh.core.properties import value
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import dodge

# Oil
output_file("oil.html")

countries = list(df['Partner.2'])
years = ['2014', '2015', '2016']

data = {'countries' : countries,
        '2014'   : df['perc_oil_2014'],
        '2015'   : df['perc_oil_2015'],
        '2016'   : df['perc_oil_2016']}

x = [ (country, year) for country in countries for year in years ]
counts = sum(zip(data['2014'], data['2015'], data['2016']), ()) 
source = ColumnDataSource(data=data)
p = figure(x_range=countries, y_range=(0, 0.5), plot_height=250, title="EU imports of crude oil by partners (%)",
           toolbar_location=None, tools="")

p.vbar(x=dodge('countries', -0.25, range=p.x_range), top='2014', width=0.2, source=source,
       color="#c9d9d3", legend=value("2014"))
p.vbar(x=dodge('countries',  0.0,  range=p.x_range), top='2015', width=0.2, source=source,
       color="#718dbf", legend=value("2015"))
p.vbar(x=dodge('countries',  0.25, range=p.x_range), top='2016', width=0.2, source=source,
       color="#e84d60", legend=value("2016"))

p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None
p.legend.location = "top_center"
p.legend.orientation = "horizontal"
show(p)


# Gas

output_file("gas.html")

countries = list(df['Partner.1'].dropna(how='any'))
years = ['2014', '2015', '2016']

data = {'countries' : countries,
        '2014'   : df['perc_gas_2014'],
        '2015'   : df['perc_gas_2015'],
        '2016'   : df['perc_gas_2016']}

x = [ (country, year) for country in countries for year in years ]
counts = sum(zip(data['2014'], data['2015'], data['2016']), ()) 
source = ColumnDataSource(data=data)
p = figure(x_range=countries, y_range=(0, 0.5), plot_height=250, title="EU imports of natural gas by partners (%)",
           toolbar_location=None, tools="")

p.vbar(x=dodge('countries', -0.25, range=p.x_range), top='2014', width=0.2, source=source,
       color="#c9d9d3", legend=value("2014"))
p.vbar(x=dodge('countries',  0.0,  range=p.x_range), top='2015', width=0.2, source=source,
       color="#718dbf", legend=value("2015"))
p.vbar(x=dodge('countries',  0.25, range=p.x_range), top='2016', width=0.2, source=source,
       color="#e84d60", legend=value("2016"))

p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None
p.legend.location = "top_center"
p.legend.orientation = "horizontal"
show(p)


#Coal

output_file("coal.html")

countries = list(df['Partner'].dropna(how='any'))
years = ['2014', '2015', '2016']

data = {'countries' : countries,
        '2014'   : df['perc_coal_2014'],
        '2015'   : df['perc_coal_2015'],
        '2016'   : df['perc_coal_2016']}

x = [ (country, year) for country in countries for year in years ]
counts = sum(zip(data['2014'], data['2015'], data['2016']), ()) 
source = ColumnDataSource(data=data)
p = figure(x_range=countries, y_range=(0, 0.5), plot_height=250, title="EU imports of solid fuel (coal) by partners (%)",
           toolbar_location=None, tools="")

p.vbar(x=dodge('countries', -0.25, range=p.x_range), top='2014', width=0.2, source=source,
       color="#c9d9d3", legend=value("2014"))
p.vbar(x=dodge('countries',  0.0,  range=p.x_range), top='2015', width=0.2, source=source,
       color="#718dbf", legend=value("2015"))
p.vbar(x=dodge('countries',  0.25, range=p.x_range), top='2016', width=0.2, source=source,
       color="#e84d60", legend=value("2016"))

p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None
p.legend.location = "top_center"
p.legend.orientation = "horizontal"
show(p)



# Energy dependence

file = 'energy_dependence.xls'
df = pd.read_excel(file)
print(df.columns)

countries = list(df['countries'].dropna(how='any'))[0:38]
years = ['2000', '2015']

data = {'countries' : countries,
        '2000'   : df['2000'],
        '2015'   : df['2015']}

x = [ (country, year) for country in countries for year in years ]
counts = sum(zip(data['2000'], data['2015']), ()) 
source = ColumnDataSource(data=data)
p = figure(x_range=countries, y_range=(-50, 120), plot_height=350, title="Energy dependence per country (all products) (%)",
           toolbar_location=None, tools="")

p.vbar(x=dodge('countries', -0.25, range=p.x_range), top='2000', width=0.25, source=source,
       color="#e84d60", legend=value("2000"))
p.vbar(x=dodge('countries',  0.0,  range=p.x_range), top='2015', width=0.25, source=source,
       color="#718dbf", legend=value("2015"))

p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = 1
p.xgrid.grid_line_color = None
p.legend.location = "top_left"
p.legend.orientation = "horizontal"
show(p)















