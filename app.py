import streamlit as st
import joblib
import json
from churn_pipeline import predict_customer

# Load the model
model = joblib.load('model.pkl')
dv = joblib.load('dv.pkl')

input_dict = {}

st.title('Churn Prediction App For Telco Customers')

input_dict['gender'] = st.radio('Gender', ('male', 'female'))
#st.write(input_dict['gender'])

seniorcitizen = st.toggle('Is the customer a senior citizen?')
#st.write(seniorcitizen)
input_dict['seniorcitizen'] = 1 if seniorcitizen else 0
#st.write(input_dict['seniorcitizen'])

partner = st.toggle('Does the customer have a partner?')
input_dict['partner'] = 'yes' if partner else 'no'

dependents = st.toggle('Does the customer have dependents?')
input_dict['dependents'] = 'yes' if dependents else 'no'

input_dict['tenure'] = st.number_input('Tenure in months', min_value=0, max_value=72, value=0)


phoneservice = st.toggle('Does the customer have phone service?')
input_dict['phoneservice'] = 'yes' if phoneservice else 'no'

multiplelines = st.toggle('Does the customer have multiple lines?')
input_dict['multiplelines'] = 'yes' if multiplelines else 'no'

input_dict['internetservice'] = st.selectbox('Internet Service', ('dsl', 'fiber_optic', 'no'))

onlinesecurity = st.toggle('Does the customer have online security?')
input_dict['onlinesecurity'] = 'yes' if onlinesecurity else 'no'

onlinebackup = st.toggle('Does the customer have online backup?')
input_dict['onlinebackup'] = 'yes' if onlinebackup else 'no'

deviceprotection = st.toggle('Does the customer have device protection?')
input_dict['deviceprotection'] = 'yes' if deviceprotection else 'no'    

techsupport = st.toggle('Does the customer have tech support?')  
input_dict['techsupport'] = 'yes' if techsupport else 'no'

streamingtv = st.toggle('Does the customer have streaming TV?')
input_dict['streamingtv'] = 'yes' if streamingtv else 'no' 

streamingmovies = st.toggle('Does the customer have streaming movies?') 
input_dict['streamingmovies'] = 'yes' if streamingmovies else 'no'

input_dict['contract'] = st.selectbox('Contract', ('month_to_month', 'one_year', 'two_year'))

paperlessbilling = st.toggle('Does the customer have paperless billing?')
input_dict['paperlessbilling'] = 'yes' if paperlessbilling else 'no'

input_dict['paymentmethod'] = st.selectbox('Payment Method', ('bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check', 'mailed_check'))


input_dict['monthlycharges'] = st.number_input('Monthly Charges', min_value=0.0, max_value=100.0, value=0.0, step=0.01, format="%.2f")

input_dict['totalcharges'] = input_dict['tenure'] * input_dict['monthlycharges']
st.write(f'Total Charges (calculated): {input_dict["totalcharges"]}')

if st.button('Predict Churn'):
    proba, pred = predict_customer(input_dict, model, dv)
    churn_prob = proba[1]
    st.metric('Churn Probability', f'{churn_prob:.4f}')
    if pred[0] == 1:
        st.warning('This customer is likely to churn')
    else:
        st.success('This customer is not likely to churn')