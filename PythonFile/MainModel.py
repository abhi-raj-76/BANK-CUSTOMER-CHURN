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
# Investigate all the elements within each feature

for column in raw_data:
    unique_vals = np.unique(raw_data[column].fillna('0'))
    nr_values = len(unique_vals)
    if nr_values <= 12:
        print('The number of values for feature {} : {} -- {}'.format(column,nr_values,unique_vals))
    else:
        print('The number of values for feature {} : {}'.format(column, nr_values))
