import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score

import streamlit as st

# streamlit containers
header = st.container()
output = st.container()

# streamlit body
with header:
    st.title("Classification Algorithm")
    st.text("")


df_le_class = pd.read_csv("dataset_classify.csv")

## Naive Bayes Classifier

y_NB = df_le_class['getDrinks']
X_NB = df_le_class[['TimeSpent_minutes', 'Hour', 'Temp_celsius', 'Wind_kmph', 'Humidity_percent']]
    
X_train_NB, X_test_NB, y_train_NB, y_test_NB = train_test_split(X_NB, y_NB, test_size=0.2, random_state=10)

# Model Creation

nb = GaussianNB()
nb.fit(X_train_NB, y_train_NB)
nb_pred = nb.predict(X_test_NB)

# Calculate the overall accuracy on test set 
a = ("Accuracy on test set: {:.3f}".format(nb.score(X_test_NB, y_test_NB)))

#Calculate AUC
prob = nb.predict_proba(X_test_NB)
prob = prob[:,1]
auc = roc_auc_score(y_test_NB, prob) 
b = ('AUC: %.2f' % auc)

fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test_NB, prob) # fpr=false positive rate, tpr=true positive rate

plt.figure(figsize = (10,5))
figNB = sns.lineplot(fpr_NB, tpr_NB, color='blue', label='NB') 
figNB = sns.lineplot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()




## Random Forest Classifier

y_RF = df_le_class['partOfDay']
X_RF = df_le_class[['Humidity_percent', 'Wind_kmph', 'Weather', 'Temp_celsius', 'Laundry_count',
                    'Age_Range', 'TimeSpent_minutes']]

X_train_RF, X_test_RF, y_train_RF, y_test_RF = train_test_split(X_RF, y_RF, test_size=0.2, random_state=10)

# Model Creation

rf = RandomForestClassifier(max_depth = 3, random_state = 10)
rf.fit(X_train_RF, y_train_RF)
y_pred = rf.predict(X_test_RF)

# Calculate the overall accuracy on test set 
c = ("Accuracy on test set: {:.3f}".format(rf.score(X_test_RF, y_test_RF)))

# Calculate AUC
# your codes here... 
prob = rf.predict_proba(X_test_RF)
prob = prob[:,1]
auc = roc_auc_score(y_test_RF, prob) 
d = ('AUC: %.2f' % auc)

fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test_RF, y_pred) # fpr=false positive rate, tpr=true positive rate

plt.figure(figsize = (10,5))
figRF = sns.lineplot(fpr_RF, tpr_RF, color='red', label='RF') 
figRF = sns.lineplot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()

with output:
    st.header("Naive Bayes")
    st.text("Determine if customers will get a drink or not depending on time spent, time of the day and weather of the day. Use to properly plan vending machines and drink stations.")
    st.write(a)
    st.write(b)
    st.pyplot(fig=figNB.figure, clear_figure=None)

    st.header("Random Forest")
    st.text("Determine which part of the day most customers will visit the laundry mart, depending on weather conditions, number of laundry marts, customer age and total time spent. Use for better staff management.")
    st.write(c)
    st.write(d)
    st.pyplot(fig=figRF.figure, clear_figure=None)
