import pandas as pd
import numpy as np
import random
import os
import sys
import requests
import time
import datetime as dt

from dotenv import load_dotenv
from flipside import Flipside
from prophet import Prophet

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from scripts.utils import flipside_api_results, set_random_seed

class EnsemblePredictor:
    def __init__(self, prophet_model, rf_model, ridge_model, features):
        self.prophet_model = prophet_model
        self.rf_model = rf_model
        self.ridge_model = ridge_model
        self.features = features

    def predict(self, X, df_prophet):
        # Prepare the input for Prophet
        future = X[['ds']].copy()
        for feature in self.features:
            future[feature] = X[feature].values
        
        forecast = self.prophet_model.predict(future)
        prophet_preds = forecast['yhat'].values
        
        # Prepare the input for RandomForest and Ridge
        X_rf = X.drop(columns=['ds'])  # Drop 'ds' for RF and Ridge
        rf_preds = self.rf_model.predict(X_rf)
        ridge_preds = self.ridge_model.predict(X_rf)

        # Create a DataFrame to store predictions
        predictions = pd.DataFrame({
            'prophet': prophet_preds,
            'rf': rf_preds,
            'ridge': ridge_preds
        })

        # Aggregate predictions (you can use mean, median, or other methods)
        predictions['ensemble'] = predictions.mean(axis=1)
        
        return predictions['ensemble']
    
class Prophet_Domain_Valuator():
    def __init__(self, domain, features, features_data):
        self.domain = domain
        self.features = features
        self.features_data = features_data
        self.data = None
    
    def model_prep(self):
        # Prepare the domain DataFrame
        domain_df = pd.DataFrame({'domain': [self.domain]})
        domain_df['domain_length'] = domain_df['domain'].apply(len)
        domain_df['num_vowels'] = domain_df['domain'].apply(lambda x: sum([1 for char in x if char in 'aeiou']))
        domain_df['num_consonants'] = domain_df['domain'].apply(lambda x: sum([1 for char in x if char.isalpha() and char not in 'aeiou']))
        domain_df['tld'] = domain_df['domain'].apply(lambda x: x.split('.')[-1])

        # Include today’s date
        today = pd.Timestamp.now().normalize()
        print(f'Domain DataFrame Columns: {domain_df.columns}')
        print(f'Feature Data (latest entry): {self.features_data.iloc[-1]}')

        # Prepare features with missing ones from features_data
        missing_features = [feature for feature in self.features if feature not in domain_df.columns]
        if missing_features:
            for feature in missing_features:
                if feature in self.features_data.columns:
                    domain_df[feature] = self.features_data[feature].iloc[-1]
                else:
                    raise ValueError(f"Feature {feature} is missing from features_data")
        
        # Ensure all features are present
        all_features = [feature for feature in self.features if feature != 'ds']
        domain_features = domain_df[all_features].iloc[0]
        
        self.data = pd.DataFrame({
            'ds': [today],
            **domain_features.to_dict()
        })

        print(f"Prepared data for prediction: {self.data}")

    def value_domain(self, model):
        self.model = model
        # Ensure the feature data includes the latest information
        future = self.data.copy()
        for feature in self.features:
            if feature not in future.columns:
                if feature in self.features_data.columns:
                    future[feature] = self.features_data[feature].iloc[-1]
                else:
                    raise ValueError(f"Feature {feature} is missing from features_data")

        # Predict using the fitted model
        forecast = self.model.predict(future)
        value = forecast['yhat'].values[0]
        print(f'Domain: {self.domain} \nPredicted value: {value}')
        return value
    
class Domain_Valuator():
    def __init__(self, domain, X, y, features, features_data, seed):
        self.domain = domain
        self.features = features
        self.features_data = features_data
        self.model = None
        self.data = None
        self.X = X
        self.y = y 
        self.seed = seed

    def model_prep(self):
        # Prepare the domain DataFrame
        domain_df = pd.DataFrame({'domain': [self.domain]})
        domain_df['domain_length'] = domain_df['domain'].apply(len)
        domain_df['num_vowels'] = domain_df['domain'].apply(lambda x: sum([1 for char in x if char in 'aeiou']))
        domain_df['num_consonants'] = domain_df['domain'].apply(lambda x: sum([1 for char in x if char.isalpha() and char not in 'aeiou']))
        domain_df['tld'] = domain_df['domain'].apply(lambda x: x.split('.')[-1])
        # domain_df['tld_weight'] = domain_df['tld'].map(self.tld_weights).fillna(self.default_tld_weight)

        # Include today’s date
        today = pd.Timestamp.now().normalize()
        print(f'Domain DataFrame Columns: {domain_df.columns}')
        print(f'Feature Data (latest entry): {self.features_data.iloc[-1]}')

        # Prepare features with missing ones from features_data
        missing_features = [feature for feature in self.features if feature not in domain_df.columns]
        print(f'domain df before adding features {domain_df}')
        print(f'domain df cols {domain_df.columns}')
        print(f'missing features {missing_features}')
        if missing_features:
            for feature in missing_features:
                if feature in self.features_data.columns:
                    domain_df[feature] = self.features_data[feature].iloc[-1]
                else:
                    raise ValueError(f"Feature {feature} is missing from features_data")
        
        # Ensure all features are present
        all_features = [feature for feature in self.features if feature != 'ds']
        domain_features = domain_df[all_features].iloc[0]
        print(f'domain features {domain_features}')
        
        self.data = pd.DataFrame({
            'ds': [today],
            **domain_features.to_dict()
        })

        print(f"Prepared data for prediction: {self.data}")
        print(f"Prepared col for prediction: {self.data.columns}")

    def value_domain(self, model):
        self.model = model
        print(f'data: {self.data}')
        domain_x = self.data[self.features]
        value = self.model.predict(domain_x)
        print(f'domain: {self.domain} \npredicted value: {value[0]}')
        return value[0]
    
def train_ridge_model(X, y, features, seed):

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), ['domain_length', 'num_vowels', 'num_consonants']),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), ['tld'])
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', Ridge(alpha=1000.0))  # Set the best alpha value from grid search
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R²: {r2}')

    return pipeline, features 



# %%
def train_randomforest_model(X, y, features, seed):

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), ['domain_length', 'num_vowels', 'num_consonants']),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), ['tld'])
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=seed))  # Set the best alpha value from grid search
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'R²: {r2}')

    return pipeline, features 

# %%
def train_prophet_model(features, combined_dataset, seed):
    df_prophet = combined_dataset.copy()
    print(df_prophet.columns)
    df_prophet.rename(columns={"dt": "ds", "price_usd": "y"}, inplace=True)    
    df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)

    target = 'y'
    features = df_prophet[features].select_dtypes(include=[np.number]).columns

    train_df, test_df = train_test_split(df_prophet, test_size=0.2, shuffle=False, random_state=seed)

    model = Prophet()

    for feature in features:
        model.add_regressor(feature)

    model.fit(train_df)

    future = test_df[['ds']].copy()
    for feature in features:
        future[feature] = test_df[feature].values

    forecast = model.predict(future)

    y_true = test_df['y'].values
    y_pred = forecast['yhat'].values

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"R²: {r2}")
    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    return model, features