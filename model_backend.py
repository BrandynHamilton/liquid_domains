from flask import Flask, render_template, request, jsonify
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import plotly.io as pio
from plotly.utils import PlotlyJSONEncoder
import json
import plotly.graph_objs as go

from diskcache import Cache
import pytz

from decimal import Decimal, ROUND_DOWN
from dotenv import load_dotenv

import requests
import time


import logging

from scripts import Prophet_Domain_Valuator, ridge_domain_valuator, randomforrest_domain_valuator, EnsemblePredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model():

prophet_model, prophet_features = train_prophet_model(prophet_features)

# %%
ridge_model, ridge_features = train_ridge_model(X, y)

# %%
randomforest_model, random_forest_features = train_randomforest_model(X, y)

prophet_valuator = Prophet_Domain_Valuator(domain, prophet_features, prophet_features_data)

# Prepare the model and get the domain value
prophet_valuator.model_prep()
prophet_domain_value = prophet_valuator.value_domain(prophet_model)

ridge_valuator = Domain_Valuator(domain, X, y, gen_features, features_data, seed)

# Prepare the model and get the domain value
ridge_valuator.model_prep()
ridge_domain_value = ridge_valuator.value_domain(ridge_model)

randomforest_valuator = Domain_Valuator(domain, X, y, gen_features, features_data, seed)

# Prepare the model and get the domain value
randomforest_valuator.model_prep()
randomforest_domain_value = randomforest_valuator.value_domain(randomforest_model)

individual_predictions = [
    prophet_domain_value,
    ridge_domain_value,
    randomforest_domain_value
]

ensemble_domain_value = np.mean(individual_predictions)

scheduler = BackgroundScheduler()

def create_app():
    app = Flask(__name__)

    def fetch_and_cache_data():
        with app.app_context():
            logger.info("Scheduled task running.")
            print("Scheduled task running.")
            latest_data()

    scheduler.add_job(
        fetch_and_cache_data, 
        trigger=CronTrigger(minute='0'),  # Ensures it runs at the top of every hour
        id='data_fetch_job',             # A unique identifier for this job
        replace_existing=True            # Ensures the job is replaced if it's already running when you restart the app
    )

    def latest_data(domain_valuation_model):
        domain_valuation_model.model_prep()
        domain_value = domain_valuation_model.value_domain(domain_valuation_model)

        