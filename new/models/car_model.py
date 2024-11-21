import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('data/used_cars.csv', encoding='ISO-8859-1')
df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
df['accident'] = df['accident'].astype(str).apply(lambda x: 1 if 'accident' in x else 0)
df.drop(columns=['int_col'], inplace=True)

X = df.drop(columns=['price'])
y = df['price']

categorical_features = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'clean_title']
numerical_features = ['model_year', 'accident']

numeric_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = RandomForestRegressor(n_estimators=100, random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

def predict_car_price(brand, model, model_year, fuel_type, engine, transmission, ext_col, accident, clean_title):
    user_input = pd.DataFrame({
        'brand': [brand],
        'model': [model],
        'model_year': [model_year],
        'fuel_type': [fuel_type],
        'engine': [engine],
        'transmission': [transmission],
        'ext_col': [ext_col],
        'accident': [1 if accident == 'At least 1 accident or damage reported' else 0],
        'clean_title': [clean_title]
    })
    predicted_price = pipeline.predict(user_input)
    return predicted_price[0]
