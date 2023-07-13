import os
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('E:\Projects\Flanker_DevMIND1_MSIT\Manuscripts\Testosterone\devmind_flanker_testosterone')

df = pd.read_csv('DevMIND_Flanker_Hormone_Preprocessing_v3.csv')

df = df[df['Excluded'] == 'N']
df = df[df['T_mean'] > 0]

age_testosterone = scipy.stats.pearsonr(df['age'], df['T_mean'])

plt.figure()
plt.scatter(df['T_mean'], df['age'])
plt.show()

df['RT_avg'] = (df['Total_Control']*df['RT_Control'] + df['Total_Flanker']*df['RT_Flanker'])/(df['Total_Control'] + df['Total_Flanker'])

testosterone_behavior = scipy.stats.pearsonr(df['RT_avg'], df['T_mean'])

plt.figure()
plt.scatter(df['T_mean'], df['RT_avg'])
plt.show()

