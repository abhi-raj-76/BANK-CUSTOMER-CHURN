import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# pip install xgboost
# pip install graphviz

my_raw_data = pd.read_csv("Churn_Modelling.csv",encoding="latin1")

# Displaying the 1st 5 row
print(my_raw_data.head())

# Understanding Our data
# Retriving all the data for given column if unique values is lesser than 12
# otherwise just give the column name and total unique value count

for column in my_raw_data:
    unique_vals = np.unique(my_raw_data[column].fillna('0'))
    nr_values = len(unique_vals)
    if nr_values <= 12:
        print('The number of values for feature {} : {} -- {}'.format(column,nr_values,unique_vals))
    else:
        print('The number of values for feature {} : {}'.format(column, nr_values))

# Now check for null values in all column

print(my_raw_data.isnull().sum()) # sum() treat true as 1 and false as 0 


figr = plt.figure(figsize=(8, 6)) # Creates a new figure object for the plot
sns.countplot(data=my_raw_data, x="Exited") # Seaborn function that creates a bar chart where, data: which data frame and x: Specify column 
plt.title("Count plot of Exited", fontsize=16) # Add the titles at the top of the plot
plt.xlabel("Exited",fontsize=14) # Gives the X label/name/title
plt.ylabel("Count",fontsize=14) # Gives Y label/name/title
plt.xticks([0, 1],labels=["Not Exited","Exited"]) # X-axis, each bar label.By default 0 and 1 because Exited column contains these two values only
plt.show()

#st.pyplot(figr) # Showing plot on streamlit 

# Column selection inside [...] this list we have to specify which columns we want to keep
# Now my new DataFrame is raw_data_v which only contains these 11 columns
raw_data_v = my_raw_data[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                     'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                     'EstimatedSalary', 'Exited']]

# Create a grid of scatter plots for each pair of numericals columns
# hue='Exited', one color for 0 and another for 1
g = sns.pairplot(raw_data_v, hue='Exited')

st.pyplot(g.figure) # We need to change the pairgrid object to figure object for showing on streamlit
