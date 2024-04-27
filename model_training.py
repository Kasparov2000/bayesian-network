import pandas as pd
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator


# Function to bin numeric data
def binning(data, column, bins, labels):
    data[column] = pd.cut(data[column], bins=bins, labels=labels, include_lowest=True, right=False)


# Load and preprocess data
data = pd.read_excel("life-insurance-data.xlsx")
data.drop(columns=["Client ID", "Policy Number", "Date of Default"], inplace=True)

# Preprocess categorical data
categorical_cols = ['Gender', 'Marital Status', 'Employment Status', 'Tier', 'Location (Urban/Rural)']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Parse money values
money_columns = ['Monthly Income (USD)', 'Monthly Premium', 'Dependants']
for col in money_columns:
    data[col] = data[col].replace('[\$,]', '', regex=True).astype(float)

# Define bins and labels for binning
age_bins = [0, 30, 50, float('inf')]
age_labels = ["Young", "Adult", "Senior"]

credit_score_bins = [0, 600, 700, 850]
credit_score_labels = ["Low", "Medium", "High"]

monthly_income_bins = [0, 200, 500, 1000, float('inf')]
monthly_income_labels = ["Low", "Medium", "High", "Very High"]

monthly_premium_bins = [0, 200, 600, 1000, float('inf')]
monthly_premium_labels = ["Low", "Medium", "High", "Very High"]

inflation_bins = [0, 20, 40, 60, 1000]
inflation_labels = ["Low_Inflation", "Medium_Low_Inflation", "Medium_High_Inflation", "High_Inflation"]

# Apply binning
binning(data, 'Age', age_bins, age_labels)
binning(data, 'Credit Score', credit_score_bins, credit_score_labels)
binning(data, 'Monthly Income (USD)', monthly_income_bins, monthly_income_labels)
binning(data, 'Monthly Premium', monthly_premium_bins, monthly_premium_labels)
binning(data, 'Monthly Inflation Rate (%)', inflation_bins, inflation_labels)

# Split the dataset into features (X) and target variable (y)
X = data.drop(columns=['Default Status'])  # Features
y = data['Default Status']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define Bayesian network structure
model = BayesianNetwork([
    ('Age', 'Default Status'),
    ('Gender_Male', 'Default Status'),
    ('Marital Status_Single', 'Default Status'),
    ('Monthly Income (USD)', 'Monthly Premium'),
    ('Monthly Income (USD)', 'Default Status'),
    ('Dependants', 'Default Status'),
    ('Marital Status_Widowed', 'Default Status'),
    ('Credit Score', 'Default Status'),
    ('Employment Status_Self-employed', 'Default Status'),
    ('Tier_Gold', 'Default Status'),
    ('Location (Urban/Rural)_Urban', 'Default Status'),
    ('Monthly Inflation Rate (%)', 'Monthly Premium'),
    ('Monthly Inflation Rate (%)', 'Default Status'),
    ('Location (Urban/Rural)_Urban', 'Default Status')
])

# Combine features and target variable for training
train_data = X_train.copy()
train_data['Default Status'] = y_train

# Train the model
model.fit(train_data, estimator=BayesianEstimator, prior_type='BDeu')
