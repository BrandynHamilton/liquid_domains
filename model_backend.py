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

prophet_model =
rf_model =
ridge_model =
prophet_features =

domain_valuation_model = EnsemblePredictor(prophet_model, rf_model, ridge_model, prophet_features)

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

        