import os
import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir('E:\Projects\Flanker_DevMIND1_MSIT\Manuscripts\Testosterone\devmind_flanker_testosterone')

df = pd.read_csv('DevMIND_Flanker_Hormone_Preprocessing_v3.csv')

df = df[df['Excluded'] == 'N']
df = df[df['T_mean'] > 0]

########################################################################################################################
# Behavioral Results - Summary Stats

np.mean(df['Accuracy_Control'])
np.std(df['Accuracy_Control'])

np.mean(df['Total_Flanker'])
np.std(df['Total_Flanker'])

np.mean(df['RT_Control'])
np.std(df['RT_Control'])

np.mean(df['RT_Flanker'])
np.std(df['RT_Flanker'])

########################################################################################################################
# Behavioral Results - Condition-wise Testing

scipy.stats.ttest_rel(df['Accuracy_Control'], df['Total_Flanker'])
scipy.stats.ttest_rel(df['RT_Flanker'], df['RT_Control'])

########################################################################################################################
# Behavioral Results - Correlations with Testosterone

age_testosterone = scipy.stats.pearsonr(df['T_mean'], df['age'])

m, b = np.polyfit(df['T_mean'], df['age'], 1)

plt.figure()
plt.scatter(df['T_mean'], df['age'])
plt.plot(df['T_mean'], m*df['T_mean']+b)
plt.show()

# Control for the effect of age on testosterone

df['RT_avg'] = (df['Accuracy_Control']*100*df['RT_Control'] + df['Total_Flanker']*df['RT_Flanker'])/(df['Accuracy_Control']*100 + df['Total_Flanker'])

testosterone_RT = scipy.stats.pearsonr(df['RT_avg'], df['T_log'])

m, b = np.polyfit(df['T_log'], df['RT_avg'], 1)

plt.figure()
plt.scatter(df['T_log_control_age'], df['RT_avg'])
plt.plot(df['T_log_control_age'], m*df['T_log_control_age']+b)
plt.show()

df['Acc_avg'] = (df['Accuracy_Control']+df['Total_Flanker'])/200

testosterone_acc = scipy.stats.pearsonr(df['Acc_avg'], df['T_residual'])

m, b = np.polyfit(df['T_residual'], df['Acc_avg'], 1)

plt.figure()
plt.scatter(df['T_residual'], df['Acc_avg'])
plt.plot(df['T_residual'], m*df['T_residual']+b)
plt.show()

testosterone_interference = scipy.stats.pearsonr(df['RT_int'], df['T_residual'])

m, b = np.polyfit(df['T_residual'], df['RT_int'], 1)

plt.figure()
plt.scatter(df['T_residual'], df['RT_int'])
plt.plot(df['T_residual'], m*df['T_residual']+b)
plt.show()

########################################################################################################################
# MEG Sensor-Level Results

scipy.stats.ttest_rel(df['Accepted_Control'], df['Accepted_Flanker_Adjusted'])
scipy.stats.pearsonr(df['Accepted_Total_Trials'], df['T_mean'])

np.mean(df['Accepted_Control'])
np.std(df['Accepted_Control'])

np.mean(df['Accepted_Flanker_Adjusted'])
np.std(df['Accepted_Flanker_Adjusted'])

np.mean(df['Accepted_Total_Trials'])
np.std(df['Accepted_Total_Trials'])


########################################################################################################################
# Theta Testosterone Correlation

os.chdir('E:\Projects\Flanker_DevMIND1_MSIT\derivatives_Testosterone\Alpha')

df = pd.read_csv('subtraction_map_correlation_T_age_regressedOut_38_-52p5_37p5.csv')

m, b = np.polyfit(df['T_residual'], df['value'], 1)

plt.figure()
plt.scatter(df['T_residual'], df['value'])
plt.plot(df['T_residual'], m*df['T_residual']+b)
plt.show()


########################################################################################################################
# Alpha Testosterone Correlation


########################################################################################################################
# Gamma Testosterone Correlation


########################################################################################################################
# Theta Age by Sex Interaction

# sns.lmplot(x='T_mean', y='value', hue='sex_0male_1female', data=df)
#
# min_val = df.describe()['connectivity']['mean'] - 3*df.describe()['connectivity']['std']
# max_val = df.describe()['connectivity']['mean'] + 3*df.describe()['connectivity']['std']
# df = df[(df['connectivity'] < max_val) & (df['connectivity'] > min_val)]
# sns.lmplot(x='age', y='connectivity', hue='sex', data=df)
# plt.xlabel('Age')
# plt.ylabel('Connectivity Interference')
# plt.savefig('E:/Projects/Flanker_DevMIND1_MSIT/Manuscripts/Development/figures/alpha_SEF_right_TPJ_FisherZ.png', dpi=300)

########################################################################################################################
# Alpha Age by Sex Interaction

########################################################################################################################
# Gamma Age by Sex Interaction








