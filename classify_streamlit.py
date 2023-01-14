import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from apyori import apriori
from sklearn import preprocessing
import streamlit as st

# streamlit containers
header = st.container()
output = st.container()

# streamlit body
with header:
    st.title("Classifcation Algorithm")
    st.text("")

with st.sidebar:
    st.header("Parameters to manipulate")

    supp_slider = st.slider(
        "Minimum support", min_value=0.0, max_value=0.035, value=0.035, step = 0.005)

    conf_slider = st.slider(
        "Minimum confidence", min_value=0.0, max_value=0.2, value=0.2)

    lift_slider = st.slider(
        "Minimum lift", min_value=2, max_value=3, value=3)

    length_slider = st.slider(
        "Minimum length", min_value=0, max_value=2, value=2)


df_le_class = pd.read_csv("dataset_classify")

y_NB = df_le_class['getDrinks']
X_NB = df_le_class[['TimeSpent_minutes', 'Hour', 'Temp_celsius', 'Wind_kmph', 'Humidity_percent']]
    
X_train_NB, X_test_NB, y_train_NB, y_test_NB = train_test_split(X_NB, y_NB, test_size=0.2, random_state=10)

# Model Creation

nb = GaussianNB()
nb.fit(X_train_NB, y_train_NB)
nb_pred = nb.predict(X_test_NB)

# Calculate the overall accuracy on test set 
print("Accuracy on test set: {:.3f}".format(nb.score(X_test_NB, y_test_NB)))

#Calculate AUC
prob = nb.predict_proba(X_test_NB)
prob = prob[:,1]
auc = roc_auc_score(y_test_NB, prob) 
print('AUC: %.2f' % auc)


y_RF = df_le_class['partOfDay']
X_RF = df_le_class[['Humidity_percent', 'Wind_kmph', 'Weather', 'Temp_celsius', 'Laundry_count',
                    'Age_Range', 'TimeSpent_minutes']]

X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(X_RF, y_RF, test_size=0.2, random_state=10)

# Model Creation

rf = RandomForestClassifier(max_depth = 3, random_state = 10)
rf.fit(X_train_RF, y_train_RF)
y_pred = rf.predict(X_test_RF)

# Calculate the overall accuracy on test set 
print("Accuracy on test set: {:.3f}".format(rf.score(X_test_RF, y_test_RF)))

# Calculate AUC
# your codes here... 
prob = rf.predict_proba(X_test_RF)
prob = prob[:,1]
auc = roc_auc_score(y_test_RF, prob) 
print('AUC: %.2f' % auc)

with output:
    st.header("Output")
    st.text("The following are the rules created based on ARM.")
    
