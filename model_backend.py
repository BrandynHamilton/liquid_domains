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
from models.forecasters import EnsemblePredictor, Prophet_Domain_Valuator, Domain_Valuator, train_ridge_model, train_randomforest_model, train_prophet_model

from vizualizations import create_visualizations

pd.options.display.float_format = '{:,.2f}'.format



API_KEY = os.getenv('FRONTEND_API_KEY')
print(f'api key: {API_KEY}')

def check_api_key(request):
    api_key = request.headers.get('Authorization')
    if api_key is None or api_key != f'Bearer {API_KEY}':
        return False
    return True

# Create Flask app
def create_app():
    app = Flask(__name__)

    cache = Cache('cache_dir')

    global historical_data
    historical_data = cache.get('historical_data', pd.DataFrame())

    # Load models
    global prophet_model, ridge_model, randomforest_model, cumulative_sales_chart, ma_plot, sold_domains_fig, rolling_avg_plot

    prophet_model = joblib.load('prophet_model.pkl')
    ridge_model = joblib.load('ridge_model.pkl')
    randomforest_model = joblib.load('randomforest_model.pkl')

    X, y, prophet_features, gen_features, target, combined_dataset, features = process_data()
    cumulative_sales_chart, ma_plot, sold_domains_fig, rolling_avg_plot = create_visualizations(combined_dataset)


    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/visualizations', methods=['GET'])
    def visualizations():
        cached_data = {
            "cumulative_sales_chart": cumulative_sales_chart,
            "ma_plot": ma_plot,
            "sold_domains_fig": sold_domains_fig,
            "rolling_avg_plot": rolling_avg_plot
        }
        return jsonify(cached_data)

    @app.route('/api/historical_data')
    def get_historical_data():
        global historical_data
        historical_data = cache.get('historical_data', pd.DataFrame())
        historical_data_json = historical_data.to_dict(orient='records')
        return jsonify(historical_data_json)

    @app.route('/api/evaluate', methods=['POST'])
    def evaluate():
        # Check for the API key
        api_key = request.headers.get('Authorization')
        if api_key != f'Bearer {API_KEY}':
            return jsonify({'error': 'Unauthorized'}), 401

        data = request.get_json()
        domain = data.get('domain')
        if not domain:
            return jsonify({'error': 'No domain provided'}), 400
        value = main(domain, prophet_model, ridge_model, randomforest_model)
        today = dt.date.today()
        value_info = {
            "dt": today,
            "domain": domain,
            "value": value
        }
        update_historical_data(value_info)
        return jsonify({'domain': domain, 'value': value})

    def update_historical_data(live_comp):
        global historical_data
        new_data = pd.DataFrame([live_comp])
        historical_data = pd.concat([historical_data, new_data]).reset_index(drop=True)
        historical_data.drop_duplicates(subset='domain', keep='last', inplace=True)
        cache.set('historical_data', historical_data)

    return app

def main(domain, prophet_model, ridge_model, randomforest_model, combined_dataset, prophet_features, gen_features, features, X, y):
    seed = 20
    set_random_seed(seed)

    graph_json_1 = json.dumps(cumulative_sales_chart, cls=PlotlyJSONEncoder)
    graph_json_2 = json.dumps(ma_plot, cls=PlotlyJSONEncoder)
    graph_json_3 = json.dumps(sold_domains_fig, cls=PlotlyJSONEncoder)
    graph_json_4 = json.dumps(rolling_avg_plot, cls=PlotlyJSONEncoder)

    prophet_features_data = combined_dataset.copy()
    prophet_features_data.rename(columns={"dt": "ds", "price_usd": "y"}, inplace=True)

    prophet_valuator = Prophet_Domain_Valuator(domain, prophet_features, prophet_features_data)
    prophet_valuator.model_prep()
    prophet_domain_value = prophet_valuator.value_domain(prophet_model)

    features_data = combined_dataset.copy()
    features_data['dt'] = features_data['dt'].dt.tz_localize(None)
    features_data = features_data[features] 

    ridge_valuator = Domain_Valuator(domain, X, y, gen_features, features_data, seed)
    ridge_valuator.model_prep()
    ridge_domain_value = ridge_valuator.value_domain(ridge_model)

    randomforest_valuator = Domain_Valuator(domain, X, y, gen_features, features_data, seed)
    randomforest_valuator.model_prep()
    randomforest_domain_value = randomforest_valuator.value_domain(randomforest_model)

    individual_predictions = [
        prophet_domain_value,
        ridge_domain_value,
        randomforest_domain_value
    ]

    ensemble_domain_value = np.mean(individual_predictions)
    print(f'individual valuations: {individual_predictions}')
    print(f'ensamble value: {ensemble_domain_value}')

    return ensemble_domain_value

if __name__ == '__main__':
    app = create_app()
    app.run(debug=False, port=5555)