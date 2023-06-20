# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 09:42:49 2023

@author: Aziza
"""
import json
import model_learner as model
import os
import os.path as op
import pandas as pd
import pickle
import data_analysis_utils as dana
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_rel

snds = ["CONST4", "RAND151", "RAND158", "RAND152", "CONST7", "RAND154", 
       "RAND153", "REG51", "RAND155", "REG56", "CONST3", "REG52", "REG53", 
       "REG58", "CONST2", "CONST5", "CONST1", "REG54", "CONST6", 
       "RAND157", "CONST8", "REG55", "REG57", "RAND156"]


# Create a function to plot scatter and linear regression
def plot_scatter_and_regression(x, y, label, color1 , color2):
    # Create scatter plot
    plt.scatter(x, y, label=label, color=color1)

    # Perform linear regression
    regression = LinearRegression()
    regression.fit(x.reshape(-1, 1), y)
    y_pred = regression.predict(x.reshape(-1, 1))

    # Plot the regression line
    plt.plot(x, y_pred, color=color2, linewidth=2)

mean_learning_rates = {}  # Dictionary to store the results


const_data_learning = []
const_data_uncertainty = []
reg5_data_learning = []
reg5_data_uncertainty = []
rand15_data_learning = []
rand15_data_uncertainty = []

for snd in snds :
    data = dana.load_data(f"C:/Users/Aziza/Stage M2 Comp brain/arousal_testing/ada-prob-participant1/ada-prob_subject-default_run-{snd}_2023-06-15.csv")
    
    session =  dana.get_sessions_per_task(data)
    
    lr_key = "learning_rate"
    array_keys = [lr_key , 'model_uncertainty']
    
    arrays = dana.get_data_arrays_from_sessions(session[0] , array_keys) 
    mean_learning_rate = np.mean(arrays['learning_rate'])
    
    mean_learning_rates[snd] = mean_learning_rate  # Store the mean learning rate in the dictionary
    
    # Extract the arrays from the data dictionary
    model_uncertainty = arrays['model_uncertainty'][0]
    learning_rate = arrays['learning_rate'][0]
    
    if snd.startswith('CONST'):
        const_data_learning.append(learning_rate)
        const_data_uncertainty.append(model_uncertainty)
    elif snd.startswith('REG5'):
        reg5_data_learning.append(learning_rate)
        reg5_data_uncertainty.append(model_uncertainty)
    elif snd.startswith('RAND15'):
        rand15_data_learning.append(learning_rate)
        rand15_data_uncertainty.append(model_uncertainty)
    
    
    plt.style.use('ggplot')
    # Create the scatter plot
    plt.scatter(model_uncertainty, learning_rate)
    plt.xlabel('Model Uncertainty')
    plt.ylabel('Learning Rate')
    plt.title('Scatter Plot: Learning Rate vs Model Uncertainty')
    
    # Perform linear regression
    regression = LinearRegression()
    regression.fit(model_uncertainty.reshape(-1, 1), learning_rate)
    predicted_learning_rate = regression.predict(model_uncertainty.reshape(-1, 1))
    
    # Plot the regression line
    plt.plot(model_uncertainty, predicted_learning_rate, color='red', linewidth=2)
    
    plt.title(snd)
    plt.show()


# Assuming you already have the dictionary mean_learning_rates

const_values = []
reg5_values = []
rand15_values = []

for key, value in mean_learning_rates.items():
    if key.startswith("CONST"):
        const_values.append(value)
    elif key.startswith("REG5"):
        reg5_values.append(value)
    elif key.startswith("RAND15"):
        rand15_values.append(value)

const_average = np.mean(const_values)
reg5_average = np.mean(reg5_values)
rand15_average = np.mean(rand15_values)

print(f"Average for CONST: {const_average}")
print(f"Average for REG5: {reg5_average}")
print(f"Average for RAND15: {rand15_average}")



t_statistics = []
p_values = []

# Paired t-test for REG RAND vs RAND CONST
t_statistic, p_value = ttest_rel(reg5_values, rand15_values)
t_statistics.append(t_statistic)
p_values.append(p_value)
print(f"Paired t-test (REG5 vs RAND15): t-statistic = {t_statistic}, p-value = {p_value}")

# Paired t-test for REG RAND vs REG CONST
t_statistic, p_value = ttest_rel(reg5_values, const_values)
t_statistics.append(t_statistic)
p_values.append(p_value)
print(f"Paired t-test (REG5 vs CONST): t-statistic = {t_statistic}, p-value = {p_value}")

# Paired t-test for RAND CONST vs REG CONST
t_statistic, p_value = ttest_rel(rand15_values, const_values)
t_statistics.append(t_statistic)
p_values.append(p_value)
print(f"Paired t-test (RAND15 vs CONST): t-statistic = {t_statistic}, p-value = {p_value}")



plt.figure()
plt.style.use('ggplot')
sns.boxplot(data=[const_values, reg5_values, rand15_values] , palette=("Paired"))
plt.xticks([0, 1, 2], ['CONST', 'REG5', 'RAND15'])
plt.xlabel('Type of auditory stimulus')
plt.ylabel('Mean Learning Rate across sessions')
plt.title('Mean Learning Rate by auditory stimulus')

# Add paired t-test results to the plot
plt.text(0.5, 1.1, f"Paired t-test (REG RAND vs RAND CONST): t-statistic = {t_statistics[0]:.2f}, p-value = {p_values[0]:.4f}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='center')
plt.text(1, 1.1, f"Paired t-test (REG RAND vs REG CONST): t-statistic = {t_statistics[1]:.2f}, p-value = {p_values[1]:.4f}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='center')
plt.text(1.5, 1.1, f"Paired t-test (RAND CONST vs REG CONST): t-statistic = {t_statistics[2]:.2f}, p-value = {p_values[2]:.4f}",
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='center')
plt.show()

# Calculate the average for const_data_learning
const_data_learning_avg = np.mean(const_data_learning, axis=0)

# Calculate the average for rand15_data_learning
rand15_data_learning_avg = np.mean(rand15_data_learning, axis=0)

# Calculate the average for reg5_data_learning
reg5_data_learning_avg = np.mean(reg5_data_learning, axis=0)

# Calculate the average for const_data_uncertainty
const_data_uncertainty_avg = np.mean(const_data_uncertainty, axis=0)

# Calculate the average for rand15_data_uncertainty
rand15_data_uncertainty_avg = np.mean(rand15_data_uncertainty, axis=0)

# Calculate the average for reg5_data_uncertainty
reg5_data_uncertainty_avg = np.mean(reg5_data_uncertainty, axis=0)



# Create a figure and axis for the plot
fig, ax = plt.subplots()
fig = plt.figure(figsize=(14, 6))
plt.style.use('ggplot')
# Plot scatter and linear regression for const
plot_scatter_and_regression(const_data_uncertainty_avg, const_data_learning_avg, 'CONST', 
                            color1='olivedrab' , color2='yellowgreen')

# Plot scatter and linear regression for reg5
plot_scatter_and_regression(reg5_data_uncertainty_avg, reg5_data_learning_avg, 'REG5', 
                            color1='royalblue' , color2='cornflowerblue')

# Plot scatter and linear regression for rand15
plot_scatter_and_regression(rand15_data_uncertainty_avg, rand15_data_learning_avg, 'RAND15', 
                            color1='palevioletred' , color2='lightpink')

# Set labels and title for the plot
plt.xlabel('Mean Uncertainty  across sessions for the same type of auditory stimulus')
plt.ylabel('Mean learning rate across sessions')
plt.title('Scatter plot and Linear Regression')

# Add legend to the plot
plt.legend()

# Show the plot
plt.show()





