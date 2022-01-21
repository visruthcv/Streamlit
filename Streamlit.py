import streamlit as st
#import streamlit_theme as stt
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

# Deasing the layout

st.title('My first Streamlit !')

st.write("""
        # This is a Heart Failure Prediction Application !
        """)

st.sidebar.header('Input Features by the User')

def user_input_features():
    anaemia = st.sidebar.selectbox('Anaemia',('No','Yes'))
    sex = st.sidebar.selectbox('Sex',('Male','Female'))
    hbp = st.sidebar.selectbox('High blood pressure',('No','Yes'))
    smoking = st.sidebar.selectbox('Smoking',('No','Yes'))
    diabetes = st.sidebar.selectbox('Diabetes',('No','Yes'))
    age = st.sidebar.slider('Age', 20,95,20)
    creatinine = st.sidebar.slider('Creatinine Phosphokinase', 23,7861,23)
    ejection = st.sidebar.slider('Ejection Fraction', 14,80,14)
    platelets = st.sidebar.slider('Platelets', 25100,850000,25100)
    serum_creatinine = st.sidebar.slider('Serum Creatinine', 0.5,9.4,0.5)
    serum_sodium = st.sidebar.slider('Serum Sodium', 113,148,113)
    lastvis = st.sidebar.slider('Follow-up period (Days)', 0,300,0)


    data = {'age': [age],
            'anaemia': [anaemia],
            'creatinine_phosphokinase': [creatinine],
            'diabetes': [diabetes],
            'ejection_fraction': [ejection],
            'high_blood_pressure': [hbp],
            'platelets': [platelets],
            'serum_creatinine': [serum_creatinine],
            'serum_sodium': [serum_sodium],
            'sex': [sex],
            'smoking': [smoking],
            'time': [lastvis]
            }
    features = pd.DataFrame(data)
    return features
input_df = user_input_features()

# Read the csv
df_raw = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df_norm = df_raw.drop(columns=['DEATH_EVENT'], axis =1)

# Encoding of ordinal features
input_df['anaemia'] = input_df['anaemia'].replace({'Yes': 1, 'No':0}).astype('int64')
input_df['high_blood_pressure'] = input_df['high_blood_pressure'].replace({'Yes': 1, 'No':0}).astype('int64')
input_df['sex'] = input_df['sex'].replace({'Male': 1, 'Female':0}).astype('int64')
input_df['smoking'] = input_df['smoking'].replace({'Yes': 1, 'No':0}).astype('int64')
input_df['diabetes'] = input_df['diabetes'].replace({'Yes': 1, 'No':0}).astype('int64')

# Displays the user input features
st.subheader('Input features')
st.write(input_df)

# Scaling
s = MinMaxScaler()
df_norm = s.fit_transform(df_norm)
input_df = s.transform(input_df)

# Reads in saved neural network model
load_clf = pickle.load(open('finalized_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)


st.subheader('Prediction')
diab_pred = np.array(['NO DEATH','DEATH'])
st.write(diab_pred[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
