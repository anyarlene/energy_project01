
#librairies
import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.pyplot as mpl
import seaborn as sns
import plotly.graph_objs as go

from datetime import datetime

import statsmodels.api as sm
from statsmodels.tsa.api import VAR

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

from scipy.stats import pearsonr
from pylab import rcParams

mpl.rcParams['figure.figsize']= (10,8)
mpl.rcParams['axes.grid']= False

import warnings
warnings.filterwarnings("ignore")


path = '/Energy project/dataset.csv'
df = pd.read_csv(path, index_col=0, parse_dates=True)
#df.head(10)

df_diff = df.diff(periods=1)
df_diff.dropna(inplace=True)

df_train = df[:int(0.98*(len(df)))]
df_test = df[int(0.98*(len(df))):]


df_diff_train = df_train.diff()
df_diff_train.dropna(inplace=True)

# check best AIC order
forecasting_model = VAR(df_diff_train)
results_aic = []
for p in range(1,50):
  results = forecasting_model.fit(p)
  results_aic.append(results.aic)


  rcParams['figure.figsize'] = 8, 4

#sns.set()
plt.plot(list(np.arange(1,50,1)), results_aic)
plt.xlabel("Order")
plt.ylabel("AIC")
plt.show()

#build model for prediction 
model = VAR(endog=df_diff_train, freq=df_diff_train.index.inferred_freq)
model_fit = model.fit(maxlags=45, ic='aic')
#print(model_fit.summary())

lag_order = model_fit.k_ar

# Input data for forecasting
input_data = df_diff_train.values[-lag_order:]
#print(input_data)
# forecasting

pred = model_fit.forecast(y=input_data, steps=len(df_test))
#input_test_data = df_test[0:10]
pred = (pd.DataFrame(pred, index=df_test.index, columns= df_diff.columns + '_pred'))
#print(pred)

#pred

# inverting transformation: transform data to orginal format
def invert_transformation(df_diff_train, pred):
    forecast = pred.copy()
    columns = df_train.columns
    for col in columns:
        forecast[str(col)+'_pred'] = df_train[col].iloc[-1] + forecast[str(col)+'_pred'].cumsum()
    return forecast
output = invert_transformation(df_diff_train, pred)
output.loc[:, ['phasor_diff1_pred']]

# combine original phasor values and predicted values
combine = pd.concat([df_test['phasor_diff1'], output['phasor_diff1_pred']], axis=1)

# Evaluation of VAR model perfomance
combine['error'] = round(combine.apply(lambda row: abs(row.phasor_diff1 - row.phasor_diff1_pred)/(row.phasor_diff1)*100 , axis = 1), 2)
combine = combine.round(decimals=6)

combine['residual'] = [combine['phasor_diff1'][i]- combine['phasor_diff1_pred'][i] for i in range(len(combine['phasor_diff1']))]
combine['Sum_residual_squared'] = combine['residual']**2

combine['averaged_error'] = combine['error']/467

combine['std_dev_n'] = combine['phasor_diff1_pred'].rolling(window=2).std()

combine['e_n'] = combine['error']/combine['std_dev_n']

#check for the standard deviation of the time series
standard_dev = combine['phasor_diff1_pred'].std()
#print('standard dev. for each step n ahead: %f' % standard_dev)


combine['a_k'] = combine['averaged_error']/standard_dev

#plot e_n and a_k for evalution of model perfomance
df_final5 = combine.reset_index()
steps3 = df_final5.head(50)

mpl.rcParams['figure.figsize']= 10,5
steps3[['e_n','a_k']].plot()

plt.ticklabel_format(style = 'plain')

plt.xlabel('steps', family='Arial', fontsize=13)
#plt.ylabel('Phasor Difference', family='Arial', fontsize=10)
#plt.xticks(rotation=45, fontsize=8)
plt.show()


# plot prediction vs actual values over time
mpl.rcParams['figure.figsize']= 10,5

steps3[['phasor_diff1','phasor_diff1_pred']].plot()

plt.xlabel('Steps', family='Arial', fontsize=13)
#plt.ylabel('Phasor Difference', family='Arial', fontsize=10)
#plt.xticks(rotation=45, fontsize=8)
plt.show()



# Plot prediction vs actual values over time with 80% & 95% confidence interval 
# Set plot size 
#from pylab import rcParams
rcParams['figure.figsize'] = 13, 5

# Plot parameters
#START_DATE_FOR_PLOTTING = '2020-09-30 00:00:00'
df_final6 = combine.copy()
steps5 = df_final6.head(20)

final_df1 = df.copy()
#other_df_copy = other_df.copy()
final_df1 = final_df1.reset_index()
time_steps = (final_df1['time'] >= ('2020-09-25 03:00:00')) & (final_df1['time'] <= ('2020-09-26 03:00:00'))
steps6 = final_df1.loc[time_steps]
steps6 = steps6.set_index('time')
#steps6.head()
#steps6 = final_df1.tail(100)

plt.plot(steps6['phasor_diff1'], color='b', label='actual values', linewidth=1.5)
#plt.plot(df_0['phasor_diff1_train'], color='orange', label='Train data', linewidth=1)
#plt.plot(test['phasor_diff1'], label='Test data', linewidth=0.7)
#plt.plot(steps5['phasor_diff1'], color='b', label='test data',linewidth=1.5, linestyle='--')
plt.plot(steps5['phasor_diff1_pred'], color='red', label='forecasted values',linewidth=1.5)


#y = np.sin(steps5['phasor_diff1_pred'])
ci = 1.960 * np.std(steps5['phasor_diff1_pred']) / np.mean(steps5['phasor_diff1_pred'])
ci2 = 1.282 * np.std(steps5['phasor_diff1_pred']) / np.mean(steps5['phasor_diff1_pred'])

#plt.plot(steps5['phasor_diff1_pred'].index, y)

plt.fill_between(steps5['phasor_diff1_pred'].index, (steps5['phasor_diff1_pred']-ci), (steps5['phasor_diff1_pred']+ci), color='orange', alpha=0.1)
plt.fill_between(steps5['phasor_diff1_pred'].index, (steps5['phasor_diff1_pred']-ci2), (steps5['phasor_diff1_pred']+ci2), color='brown', alpha=0.1)
#import seaborn as sns
#sns.lineplot(steps3['phasor_diff1'], steps5['phasor_diff1_pred'], color='blue', label='Forecasted values',linewidth=0.8)


plt.axvline(x = max(steps6.index), color='grey', linewidth=2, linestyle='--')

plt.grid(which='major', color='#cccccc', alpha=0.5)

plt.legend(shadow=True)
#plt.title('Predcitions and Acutal values', family='Arial', fontsize=12)
plt.xlabel('time', family='Arial', fontsize=13)
plt.ylabel('Phasor Difference', family='Arial', fontsize=13)
plt.xticks(rotation=45, fontsize=8)
plt.show()