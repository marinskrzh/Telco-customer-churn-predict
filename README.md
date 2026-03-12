# Telco Customer Churn Prediction

## Project Overview
This is my personal data science and machine learning project focused on predicting customer churn for a telecommunications company. Throughout the project, I conducted Exploratory Data Analysis (EDA), calculated Mutual Information metrics and Pearson correlation for the features, and trained a baseline Logistic Regression model.

This project is currently under development: I plan to add detailed model evaluation metrics (Precision, Recall, ROC-AUC) and implement cross-validation in the near future.

## Repository Structure
- `churn-predict.ipynb` — The main Jupyter Notebook containing the research process and code.
- `requirements.txt` — A list of required Python libraries and their versions for running the project.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — The original dataset (tracked in Git due to its small size).
- `.gitignore` — Git configuration file to exclude unnecessary files like virtual environments.

## Dataset
This project uses the *Telco Customer Churn* dataset. It contains information about customers, their service plans, tenure, and the target variable `Churn` — which indicates whether the customer has left the provider.
*(The dataset is relatively small, around 1 MB, so it is kept alongside the code).*

## How to run

1. Clone the repository:
   ```bash
   git clone <YOUR_REPOSITORY_URL>
   cd TelcoCustomerChurn
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # For macOS/Linux
   ```

3. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook churn-predict.ipynb
   ```

## Future Plans
1. Calculate the Confusion Matrix and ROC AUC.
2. Implement cross-validation for more robust model evaluation.
3. Develop an interactive web application based on Streamlit to showcase the model's predictions.
