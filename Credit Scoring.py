
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

data = pd.read_csv("C:\\Users\\91877\\Downloads\\CODEALPHA\\codeAlpha_ML_project\\Credit_Data.csv")


print(data.isnull().sum())

data = data.dropna()

X = data.drop('default', axis=1) 
y = data['default']  


numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),       
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)  
    ])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)


new_data = pd.DataFrame({
    'age': [35],
    'income': [60000],
    'credit_score': [750],
    'loan_amount': [10000],
    
    'personal_status_sex': ['male'],
    'housing': ['own']
})

new_data_scaled = pipeline.predict(new_data)
print(f"Predicted Creditworthiness: {'Creditworthy' if new_data_scaled[0] == 1 else 'Not Creditworthy'}")
