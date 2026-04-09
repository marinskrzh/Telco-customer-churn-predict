# Telco Customer Churn Prediction

## Project Overview

A machine learning project for predicting customer churn in a telecommunications company. The project includes full EDA, feature engineering (Mutual Information, Pearson correlation), a trained Logistic Regression model, and a deployed interactive web application.

## Repository Structure

| File | Description |
|------|-------------|
| `churn-predict.ipynb` | Jupyter Notebook with EDA, feature analysis, and model training research |
| `churn_pipeline.py` | Pipeline script: data loading, feature preparation, training, evaluation, model export |
| `app.py` | Streamlit web application for interactive churn prediction |
| `model.pkl` | Trained Logistic Regression model (serialized) |
| `dv.pkl` | Fitted DictVectorizer for feature encoding (serialized) |
| `test_customer.json` | Sample customer likely to churn (for testing) |
| `test_customer_no_churn.json` | Sample customer unlikely to churn (for testing) |
| `requirements.txt` | Full Python dependencies (local dev + notebook) |
| `requirements-app.txt` | Minimal dependencies for Streamlit Cloud deployment |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Original dataset |

## Dataset

The [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset contains information about ~7000 customers: demographics, service subscriptions, contract type, billing, and the target variable `Churn`.

## Model

- Algorithm: Logistic Regression (`C=0.1`, solver `liblinear`)
- Features: contract type, internet service, online security, tenure, monthly charges, and others (low-MI features like `gender`, `phoneservice`, `multiplelines` were dropped)
- Metrics on test set: AUC ~0.84, F1 ~0.60

## How to Run Locally

1. Clone the repository:
   ```bash
   git clone <YOUR_REPOSITORY_URL>
   cd TelcoCustomerChurn
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. (Optional) Retrain the model:
   ```bash
   python churn_pipeline.py
   ```

6. (Optional) Explore the research notebook:
   ```bash
   pip install jupyter
   jupyter notebook churn-predict.ipynb
   ```



## Streamlit Cloud Deployment

To deploy your own fork on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push the repository to GitHub (make sure `model.pkl`, `dv.pkl`, and `requirements-app.txt` are included)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo, branch `main`, entry point `app.py`, and set requirements file to `requirements-app.txt`
4. Click Deploy
