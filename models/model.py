import pandas as pd
import numpy as np
import random
import os
import sys
import requests
import time
import datetime as dt
from diskcache import Cache
import joblib


from plotly.utils import PlotlyJSONEncoder

from flask import Flask, request, jsonify, render_template

from dotenv import load_dotenv
from flipside import Flipside
from prophet import Prophet
import json

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV

from scripts.utils import flipside_api_results, set_random_seed
from scripts.data_processing import process_data
from forecasters import EnsemblePredictor, Prophet_Domain_Valuator, Domain_Valuator, train_ridge_model, train_randomforest_model, train_prophet_model

from liquid_domains.scripts.vizualizations import create_visualizations

pd.options.display.float_format = '{:,.2f}'.format

def train_model():
    seed = 20
    set_random_seed(seed)

    X, y, prophet_features, gen_features, target, combined_dataset, features = process_data()
    # Train models and save them
    prophet_model, prophet_features = train_prophet_model(prophet_features, combined_dataset, seed)
    ridge_model, ridge_features = train_ridge_model(X, y, features, seed)
    randomforest_model, random_forest_features = train_randomforest_model(X, y, features, seed)
    
    joblib.dump(prophet_model, 'prophet_model.pkl')
    joblib.dump(ridge_model, 'ridge_model.pkl')
    joblib.dump(randomforest_model, 'randomforest_model.pkl')

if __name__ == "__main__":
    train_model()