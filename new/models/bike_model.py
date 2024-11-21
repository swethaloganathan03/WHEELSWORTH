import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Load the bike dataset
df = pd.read_csv('data/Used_Bikes.csv', encoding='ISO-8859-1')

# Ensure that the 'price' column is numeric (just for safety)
df['price'] = df['price'].astype(float)

# Define the features and target
X = df.drop(columns=['price'])
y = df['price']

# List of categorical and numerical features
categorical_features = ['bike_name', 'brand']
numerical_features = ['kms_driven', 'age', 'power']

# Define preprocessors for numerical and categorical data
numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine the preprocessors into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a RandomForest pipeline
model = RandomForestRegressor(n_estimators=100, random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the pipeline model
pipeline.fit(X_train, y_train)

# Function to predict bike price
def predict_bike_price(bike_name: str, kms_driven: int, age: int, power: float, brand: str) -> float:
    # Prepare the user input as a DataFrame
    user_input = pd.DataFrame({
        'bike_name': [bike_name],
        'kms_driven': [kms_driven],
        'age': [age],
        'power': [power],
        'brand': [brand]
    })

    # Make the prediction
    predicted_price = pipeline.predict(user_input)
    
    # Return the predicted price
    return predicted_price[0]
