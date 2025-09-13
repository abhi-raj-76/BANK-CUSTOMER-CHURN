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

raw_data = pd.read_csv("Churn_Modelling.csv",encoding="latin1")

# Displaying the 1st 5 row
print(raw_data.head())

# Understanding Our data
# Retriving all the data for given column if unique values is lesser than 12
# otherwise just give the column name and total unique value count

for column in raw_data:
    unique_vals = np.unique(raw_data[column].fillna('0'))
    nr_values = len(unique_vals)
    if nr_values <= 12:
        print('The number of values for feature {} : {} -- {}'.format(column,nr_values,unique_vals))
    else:
        print('The number of values for feature {} : {}'.format(column, nr_values))

# Now check for null values in all column

print(raw_data.isnull().sum()) # sum() treat true as 1 and false as 0 


fig = plt.figure(figsize=(8, 6)) # Creates a new figure object for the plot
sns.countplot(data=raw_data, x="Exited") # Seaborn function that creates a bar chart where, data: which data frame and x: Specify column 
plt.title("Count plot of Exited", fontsize=16) # Add the titles at the top of the plot
plt.xlabel("Exited",fontsize=14) # Gives the X label/name/title
plt.ylabel("Count",fontsize=14) # Gives Y label/name/title
plt.xticks([0, 1],labels=["Not Exited","Exited"]) # X-axis, each bar label.By default 0 and 1 because Exited column contains these two values only
plt.show()

st.pyplot(fig) # Showing plot on streamlit 

