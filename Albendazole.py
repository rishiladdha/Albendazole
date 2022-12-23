# %% 1. Startup
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# %% 2. Load the data
data = pd.read_stata('albendazole.dta')
print(data.head())
data.describe()

# %% 3. Data Exploration
plt.hist(data['test98'], color='blue', edgecolor='black', bins=15)
plt.xlabel('Score')
plt.ylabel('Number of pupils')
plt.title('Test Scores in 1998')
plt.show()

# Mean, min, max, std
mean_test98 = np.mean(data['test98'])
min_test98 = np.min(data['test98'])
max_test98 = np.max(data['test98'])
std_test98 = np.std(data['test98'])
print(mean_test98)
print(min_test98)
print(max_test98)
print(std_test98)

#the np.mean() and np.std() functions return mean and std with more 
# significant digits than the mean and std obtained by the .describe() function.

#Applying Transformation - Normalisation
data['test96'] = (data['test96_a'] - data['test96_a'].mean()) / data['test96_a'].std()
#print(data['test96_a'])
data.boxplot(column=['test98','test96'])

# %% 4. Treatment effect
treated = data[data['t98']==1]
#print(treated.head())
control = data[data['t98']==0]
#print(control.head())

#boxplot comparing the two groups in terms of prs991
data.boxplot(column=['prs991'], by='t98')

#No probelms were encountered because Pandas removes NAN values before plotting, unlike Matplotlib. 
#The boxplots are as what intuition would suggest. 
# Pupils in the treatment group have a higher prs991 score than pupils in the control group, 
# which means that treatment positively affects school participation.

#Independent t-test is a suitable test for this case because the two groups are independent.

#drop NA values in treted and control
treated2 = treated.dropna(subset=['prs991'])
control2 = control.dropna(subset=['prs991'])

print(treated2.head())
print(control2.head())
result_participation = stats.ttest_ind(treated2['prs991'], control2['prs991'])
print(result_participation)

#Since the statistic is positive, treatment positively affects school participation.
#From the test conducted, we can conclude that the mean of the two groups are not equal, because the p-value is less than 0.05. 
# Therefore, we can reject the null hypothesis that the mean of the two groups are equal.

####____________####

#Paired t-test to compare the test scores of the same students in 1996 and 1998. 

dropped = treated.dropna(subset=['test96'])
dropped = dropped.dropna(subset=['test98'])

#paired t-test on test98 and test96
result_scores = stats.ttest_rel(dropped['test98'], dropped['test96'])
print(result_scores)

#From the test conducted, we can conclude that the mean of the two groups are not equal, because the p-value is less than 0.05.
# Therefore, we can reject the null hypothesis that the mean of the two groups are equal.

# %% 5. Regression
fig, ax = plt.subplots()
dropped = data[~np.isnan(data['t98'])]
dropped = dropped[~np.isnan(data['prs991'])]
# Linear Regression
result_regression = stats.linregress(dropped['t98'], dropped['prs991'], alternative="two-sided")
ax.scatter(dropped['t98'], dropped['prs991'])
plt.plot(dropped['t98'], dropped['prs991'], 'o', label='Data Points')
plt.plot(dropped['t98'], result_regression.intercept + result_regression.slope*dropped['t98'], 'r', label='Regression line')
plt.legend()
plt.show()
print(result_regression)
#This plot shows that the treatment has a positive effect on school participation.
#p-value changes when we change alternative. Default is two-sided, but we can also use 'less' or 'greater'.








