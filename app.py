import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

from tensorflow.keras.models import load_model
#load pickle file for prediction :trained model,scaler,onehot
model=load_model('churn_model.h5')
#load scalar and encoder
with open('scaler.pkl','rb') as file:
  scaler=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
  label_encoder_gender=pickle.load(file)

with open('one_hot_encoder_geo.pkl','rb') as file:
  one_hot_encoder_geo=pickle.load(file)
  
##streamlit app
import streamlit as st
#title
st.title('Churn Prediction App')
#user input
Credit_Score=st.number_input('Credit Score')
geography=st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',19,91)
tenure=st.slider('Tenure',0,10)
balance=st.number_input('Balance')
Num_OfProducts=st.slider('Number of Products',1,4)
Has_CrCard=st.selectbox('Has Credit Card',['0','1'])
Is_Active_Member=st.selectbox('Is Active Member',['0','1'])
Estimated_Salary=st.number_input('Estimated Salary')

#preparing input data
input_data=pd.DataFrame({'CreditScore': [Credit_Score],
           'Gender' :[label_encoder_gender.transform([gender])[0]] ,
           'Age' : [age],
           'Tenure' : [tenure],
           'Balance' : [balance],
           'NumOfProducts' : [Num_OfProducts],
           'HasCrCard' : [Has_CrCard],
           'IsActiveMember' : [Is_Active_Member],
           'EstimatedSalary' : [Estimated_Salary]})

geo_encoded_df=pd.DataFrame(one_hot_encoder_geo.transform([[geography]]),columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))
input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
input_data=scaler.transform(input_data)
#prediction
predicted=model.predict(input_data)
prediction_proba=predicted[0][0]

if prediction_proba>0.5 :
  st.write("The customer is likely to churn.")
else:
  st.write("The customer is likely to stay.")






