import pymannkendall as mk
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np


def get_trend(x):
    """Define a function to fit a linear regression model and return the slope, 
    intercept, and p-value."""
    model = smf.ols('value ~ timestamp', data=x).fit()
    return pd.Series([model.params['Intercept'], 
                      model.params['timestamp'], 
                      model.pvalues['timestamp']], 
                     index=['intercept', 'slope', 'p-value'])

def get_mk_trend(x):
    """Define a function to calculate the Mann-Kendall trend and return the trend 
    and p-value."""
    mk_res = mk.seasonal_test(x.value, period=12)
    return pd.Series([mk_res.intercept, mk_res.slope, mk_res.p],
                     index=['intercept', 'slope', 'p-value'])

def plot_data_and_trendline(data, color, mann_kendall=True):
    """
    Function to plot all the well data and the associated trendlines.
    
    Switch mann_kendall to False if using linear regression"""
    
    x = data['timestamp']
    y = data['value']
    slope = data['slope'].iloc[0]
    intercept = data['intercept'].iloc[0]
    p_value = data['p-value'].iloc[0]
    
    # Convert the 'time' column to datetime for plotting
    x_range = np.arange(len(x))/12
    x_datetime = pd.to_datetime(x, unit='s')
    
    # Plot the data
    plt.plot(x_datetime, y, color='gray', linewidth=0.75)
    
    # Set the color of the trendline based on the p-value
    trend_color = 'red' if p_value < 0.05 else 'black'

    # Add the p-value to the plot
    plt.text(0.3, 0.1, f'p-value: {p_value:.2e}', ha='left', va='top', transform=plt.gca().transAxes)
    plt.text(0.3, 0.05, f'slope: {slope:.2e}', ha='left', va='top', transform=plt.gca().transAxes)

    # Plot the trendline
    if mann_kendall:
        plt.plot(x_datetime, intercept + slope * x_range, color=trend_color)
    else:
        plt.plot(x_datetime, intercept + slope * x, color=trend_color)

 
def plot_data_and_quantreg(data, color, quants=[0.1, 0.9], plot=True):
    """Function to calculate and plot the data and the quantile regresion lines.
    """
    data = data.dropna()
    x = np.arange(len(data['time'])) 
    y = data['value']
    data_df = pd.DataFrame({'x': x, 'y': y})
    
    # Quantile regresssion
    model = smf.quantreg('y ~ x', data_df)
    q1 = model.fit(q=quants[0])
    q2 = model.fit(q=quants[1])
    q3 = model.fit(q=0.5)
    
    # Convert the 'time' column to datetime for plotting
    x_datetime = pd.to_datetime(data['time'], unit='s')
    
    if plot:
        # Plot the data
        plt.plot(x_datetime, y, color='gray', linewidth=0.75)
        
        # Plot the quantile regression lines
        plt.plot(x_datetime, q1.fittedvalues, color='black')
        plt.plot(x_datetime, q2.fittedvalues, color='black')
        plt.plot(x_datetime, q3.fittedvalues, '--', color='black')
    else:
        return q1, q2, q3