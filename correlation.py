
# librairies

import pandas as pd # data processing
import numpy as np # linear algebra

import matplotlib.pyplot as plt # used for graph plot
import seaborn as sns


path1 = 'Energy project/solar.csv'

path2 = 'Energy project/phasor.csv'

aggregate_pv_df = pd.read_csv(path1)
phasor_df = pd.read_csv(path2)

aggregate_pv_df['time'] = pd.to_datetime(aggregate_pv_df['time'], format='%Y-%m-%d %H:%M')

#aggregate_pv_df['Date'] = [d.date() for d in aggregate_pv_df['time']]
#aggregate_pv_df['Time'] = [d.time() for d in aggregate_pv_df['time']]

aggregate_pv_df = aggregate_pv_df.set_index(['time'])
#aggregate_pv_df.head()

energy = aggregate_pv_df.copy()
energy = energy.reset_index()
time_series = (energy['time'] >= ('2020-01-07 00:00:00')) & (energy['time'] <= ('2020-09-30 23:45:00'))
pv_df = energy.loc[time_series]
pv_df = pv_df.set_index(['time'])
#pv_df.head()


phasor_df['time'] = pd.to_datetime(phasor_df['time'], format='%Y-%m-%d %H:%M')
phasor_df = phasor_df.set_index(['time'])


# combine solar and phasor dataset and compute correlation
data1 = phasor_df.copy()
data2 = pv_df.copy()
data1 = data1.reset_index()
data2 = data2.reset_index()

all_data = data1.merge(data2, left_on=['time'], right_on=['time'], how='right')

df = all_data.copy()
df.rename(columns = {'angle1':'phasor_diff1','angle2':'phasor_diff2','angle3':'phasor_diff3',
                   'Hochrechnung_Total':'PV_hoch', 'Prognose_Total':'PV_prog'}, inplace = True)

#df.head()
#print("phasor_diff1 : Bremen_Schondorf")
#print("phasor_diff2: Herzogenrath_Schondorf")
#print("phasor_diff3: Bremen_Büdingen")


df_corr = df[['phasor_diff1', 'phasor_diff2', 'phasor_diff3', 'PV_hoch', 'PV_prog']].corr(method="pearson")


g = sns.heatmap(df_corr, vmax=0.6, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, annot=True, fmt='.2f', cmap='coolwarm')

g.figure.set_size_inches(10, 10)

plt.show()

#save new dataframe as csv
df.to_csv("mycsvfile.csv", index=False)

