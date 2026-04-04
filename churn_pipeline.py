import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import json
import joblib

# 1. load_data()        — загрузка и очистка
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # Converting to lowercase
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    str_cols = df.columns[df.dtypes == 'object'].tolist()
    for col in str_cols:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    # Converting target
    df['churn'] = (df['churn'] == 'yes').astype(int)

    return df

# 2. prepare_features() — кодирование через DictVectorizer
def prepare_features(df, dv=None, fit_vectorizer=False):
    # Categorical features
    categorical_full = df.columns[df.dtypes == 'object'].tolist()
    categorical_full.remove('customerid')
    categorical_full.append('seniorcitizen')

    # Numerical features
    numerical = df.columns[df.dtypes != 'object'].tolist()
    numerical = list(filter(lambda x: x not in ['churn', 'seniorcitizen'], numerical))
   
   # From MI analysis drop freatures with low importance
    categorical = categorical_full.copy()
    categorical.remove('multiplelines')
    categorical.remove('phoneservice')
    categorical.remove('gender')

    # One-Hot Encoding
    
    full_dict = df[categorical + numerical].to_dict(orient='records')
    y = df['churn'].values
    if fit_vectorizer:
        dv = DictVectorizer(sparse=False)
        dv.fit(full_dict)
        X = dv.transform(full_dict)
        return X, y, dv
    else:
        if dv is None:
            raise Exception('DictVectorizer is not fit yet')
        X = dv.transform(full_dict)
        return X, y, None

# 3. train()            — обучение с лучшим C=0.1
def train(X, y, C0=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = LogisticRegression(solver='liblinear', random_state=1, C=C0)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

# 4. evaluate()         — AUC + метрики
def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_prob)
    f1 = f1_score(y, y_pred)
    return {'AUC': auc, 'F1': f1}

# 5. predict_customer() — предсказание для одного клиента
def predict_customer(customer_dict, model, dv):
    X = dv.transform([customer_dict])
    return model.predict_proba(X)[0], model.predict(X)

    
# 6. main()             — всё вместе

# [Web App / API] → получает данные клиента → вызывает predict_customer() → возвращает вероятность чёрна

if __name__ == '__main__':
    df = load_data()
    X, y, dv = prepare_features(df, fit_vectorizer=True)
    C0 = 0.1
    model, _, X_test, _, y_test = train(X, y, C0)
    print(f'For model {model} with C={C0} \nPerformance: {evaluate(model, X_test, y_test)}')
    metrics = evaluate(model, X_test, y_test)

    # Prediction for one Customer

    print('Test customer with churn')
    customer_dict_churn = {}
    with open ('test_customer.json') as f:
        customer_dict = json.load(f)
    proba, pred = predict_customer(customer_dict, model, dv)
    print(f'Probability for customer is: {proba}')
    print(f'Prediction for customer is: {pred}')
    
    print()
    
    print('Test customer without churn')
    customer_dict_no_churn = {}
    with open ('test_customer_no_churn.json') as f:
        customer_dict = json.load(f)
    proba, pred = predict_customer(customer_dict, model, dv)
    print(f'Probability for customer is: {proba}')
    print(f'Prediction for customer is: {pred}')

    # Save model
    joblib.dump(model, 'model.pkl')
    joblib.dump(dv, 'dv.pkl')
