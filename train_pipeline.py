# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pickle

# Load the Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Separate features and target
X = data.drop(columns=['Survived'])
y = data['Survived']

# Define numerical and categorical features
numerical_features = ['Age', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing for numerical and categorical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
    ('scaler', StandardScaler())                   # Standardize numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # OneHotEncode categorical variables
])

# Combine preprocessors in a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Create a pipeline with preprocessing and DecisionTreeClassifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Train the pipeline
model_pipeline.fit(X_train, y_train)

# Evaluate the model
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the pipeline to a file using pickle
with open('titanic_pipeline.pkl', 'wb') as file:
    pickle.dump(model_pipeline, file)

print("Pipeline saved as 'titanic_pipeline.pkl'.")

# Load the model back to verify
with open('titanic_pipeline.pkl', 'rb') as file:
    loaded_pipeline = pickle.load(file)

# Make predictions on new data
new_data = pd.DataFrame({
    'Pclass': [3],
    'Sex': ['male'],
    'Age': [22],
    'Fare': [7.25],
    'Embarked': ['S']
})

prediction = loaded_pipeline.predict(new_data)
print(f"Prediction for new data: {prediction[0]} (1 = Survived, 0 = Not Survived)")
