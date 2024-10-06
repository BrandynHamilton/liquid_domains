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
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV

from scripts.utils import flipside_api_results, set_random_seed
from scripts.pull_data import pull_data
from models.forecasters import EnsemblePredictor, Prophet_Domain_Valuator, Domain_Valuator, train_ridge_model, train_randomforest_model, train_prophet_model
# from sql_queries.sql_scripts import three_dns_sales

# %%
pd.options.display.float_format = '{:,.2f}'.format


# %%
current_directory = os.getcwd()
current_directory

# %%
load_dotenv()

def process_data(seed=20,api=False, web2_data = False,tld_weight=dict(com=None,net=None,box=None,org=None,eth=None),temporals=True,corr_type='pearson',
                 threshold=None,):

    # %%
    set_random_seed(seed)

    

    # %%
    flipside_api_key = os.getenv('FLIPSIDE_API_KEY')
    alchemy_api_key = os.getenv('ALCHEMY_API_KEY')
    opensea_api_key = os.getenv('OPENSEA_API_KEY')

    print(alchemy_api_key)

    data_dict = pull_data(api=api)

    if web2_data == True:

        domain_path = '../data/domain-name-sales.tsv'  
        domain_data = pd.read_csv(domain_path, delimiter='\t')

        # %%
        domain_data.set_index('date', inplace=True)
        domain_data = domain_data.drop(columns=['venue'])
        domain_data.sort_index(inplace=True)
        domain_data

        domain_data = domain_data.reset_index()
        domain_data = domain_data.rename(columns={"date":"dt","price":"price_usd"})
        domain_data['dt'] = pd.to_datetime(domain_data['dt'])
        domain_data['dt'] = domain_data['dt'].dt.tz_localize('UTC')
        domain_data['dt'] = pd.to_datetime(domain_data['dt'])

    optimism_name_service_data= data_dict['optimism_name_service_data']
    optimism_name_service_data = optimism_name_service_data[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]
    optimism_name_service_data

    Three_DNS_data = data_dict['Three_DNS_data']
    Three_DNS_data = Three_DNS_data[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]
    Three_DNS_data
  
    optimistic_domains = data_dict['optimistic_domains']

    ens_data = data_dict['ens']
    ens_data = ens_data[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]

    
    unstoppable_sales_data = data_dict['unstoppable_sales_data']
    unstoppable_sales_data = unstoppable_sales_data[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]
    unstoppable_sales_data

    
    base_domains_metadata_pd = data_dict['base_domains_metadata_pd']
    base_domains_metadata_pd = base_domains_metadata_pd[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]
    base_domains_metadata_pd

    prices_data = data_dict['prices_data']
    prices_data = prices_data.dropna()
    prices_data['SYMBOL'] = prices_data['SYMBOL'].replace('WETH', 'ETH')


    prices_data = prices_data.pivot(index='DT',columns='SYMBOL',values='PRICE')
    prices_data = prices_data.reset_index()
    prices_data

    optimistic_domains_sales = data_dict['optimistic_domains_sales']

    # %%
    optimistic_domains_sales = optimistic_domains_sales.dropna(subset=['TOKENID'])
    optimistic_domains_sales['TOKENID']

    # %%
    optimistic_domains_sales['TOKENID'] = optimistic_domains_sales['TOKENID'].astype(int)
    optimistic_domains_sales.rename(columns={"TOKENID":"tokenId"}, inplace=True)

    # %%
    optimistic_domains_sales['tokenId']

    # %%
    optimistic_domains['tokenId']


    # %%
    optimistic_data = pd.merge(optimistic_domains_sales, optimistic_domains, on='tokenId', how='left')
    optimistic_data.rename(columns={"tokenId":"nft_identifier","name":"nft_name", "day":"dt"}, inplace=True)

    # %%
    prices_data

    # %%
    optimism_name_service_data['dt'] = pd.to_datetime(optimism_name_service_data['dt'], unit='ms')
    Three_DNS_data['dt'] = pd.to_datetime(Three_DNS_data['dt'], unit='ms')
    ens_data['dt'] = pd.to_datetime(ens_data['dt'])
    unstoppable_sales_data['dt'] = pd.to_datetime(unstoppable_sales_data['dt'])
    base_domains_metadata_pd['dt'] = pd.to_datetime(base_domains_metadata_pd['dt'], unit='ms')


    optimism_name_service_data

    # %%
    def hourly(df):
        df['dt'] = df['dt'].dt.strftime('%Y-%m-%d %H-00-00')
        df['dt'] = pd.to_datetime(df['dt'])
        return df


    # %%
    Three_DNS_data = hourly(Three_DNS_data)
    optimism_name_service_data = hourly(optimism_name_service_data)
    ens_data = hourly(ens_data)
    unstoppable_sales_data = hourly(unstoppable_sales_data)
    base_domains_metadata_pd = hourly(base_domains_metadata_pd)

    Three_DNS_data

    # %%
    Three_DNS_data['dt']

    # %%
    prices_data['DT'] = pd.to_datetime(prices_data['DT'])
    prices_data.rename(columns={'DT':'dt'}, inplace=True)


    # %%
    prices_data['dt'] = prices_data['dt'].dt.tz_localize('UTC')
    prices_data

    # %%
    Three_DNS_data = Three_DNS_data.merge(prices_data, how='left', on='dt')
    Three_DNS_data['price_usd'] = Three_DNS_data['token_amt_clean'] * Three_DNS_data['ETH']
    Three_DNS_data

    # %%
    optimism_name_service_data = optimism_name_service_data.merge(prices_data, how='left', on='dt')
    optimism_name_service_data['price_usd'] = optimism_name_service_data['token_amt_clean'] * optimism_name_service_data['ETH']
    optimism_name_service_data


    # %%
    ens_data = ens_data.merge(prices_data, how='left', on='dt')
    ens_data['price_usd'] = ens_data['token_amt_clean'] * ens_data['ETH']
    ens_data

    # %%
    unstoppable_sales_data = unstoppable_sales_data.merge(prices_data, how='left', on='dt')
    unstoppable_sales_data['price_usd'] = unstoppable_sales_data['token_amt_clean'] * unstoppable_sales_data['ETH']
    unstoppable_sales_data


    # %%
    base_domains_metadata_pd = base_domains_metadata_pd.merge(prices_data, how='left', on='dt')
    base_domains_metadata_pd['price_usd'] = base_domains_metadata_pd['token_amt_clean'] * base_domains_metadata_pd['ETH']
    base_domains_metadata_pd


    # %%
    optimistic_data.rename(columns={'DAY':'dt','PRICE_USD':'price_usd','PRICE':'token_amt_clean'}, inplace=True)

    # %%
    optimistic_data['dt'] = pd.to_datetime(optimistic_data['dt'])
    optimistic_data['dt'] = optimistic_data['dt'].dt.tz_localize('UTC')
    optimistic_data['dt'] = pd.to_datetime(optimistic_data['dt'])


    # %%
    base_domains_metadata_pd

    # %%
    combined_dataset = pd.concat([
        ens_data[['dt','nft_name','price_usd','token_amt_clean']].dropna(),
        optimistic_data[['dt','nft_name','price_usd','token_amt_clean']].dropna(),
        optimism_name_service_data[['dt','nft_name','price_usd','token_amt_clean']].dropna(),
        unstoppable_sales_data[['dt','nft_name','price_usd','token_amt_clean']].dropna(),
        base_domains_metadata_pd[['dt','nft_name','price_usd','token_amt_clean']].dropna(),
        Three_DNS_data[['dt','nft_name','price_usd','token_amt_clean']].dropna()
    ], ignore_index=True)

    

    combined_dataset = combined_dataset.drop_duplicates()
    combined_dataset['dt'] = pd.to_datetime(combined_dataset['dt'], errors='coerce')
    combined_dataset = combined_dataset.sort_values(by='dt')
    combined_dataset = combined_dataset.reset_index(drop=True)
    combined_dataset


    # %%

    target = 'price_usd'
    combined_dataset = combined_dataset.rename(columns={'nft_name':'domain'})
    combined_dataset = combined_dataset.drop(columns=['token_amt_clean'])
    if web2_data == True:
        combined_dataset = pd.concat([combined_dataset, domain_data], ignore_index=True)

    combined_dataset['domain_length'] = combined_dataset['domain'].apply(len)
    combined_dataset['num_vowels'] = combined_dataset['domain'].apply(lambda x: sum([1 for char in x if char in 'aeiou']))
    combined_dataset['num_consonants'] = combined_dataset['domain'].apply(lambda x: sum([1 for char in x if char.isalpha() and char not in 'aeiou']))
    combined_dataset['tld'] = combined_dataset['domain'].apply(lambda x: x.split('.')[-1])  # Extract TLD

    if tld_weight != None:
        tld_weights = {
            'com': tld_weight['com'],
            'net': tld_weight['net'],
            'org': tld_weight['org'],
            'box': tld_weight['box'],
            'eth': tld_weight['eth']
            # Add more TLDs and their weights as needed
        }
        
        default_tld_weight = 1
        
        combined_dataset['tld_weight'] = combined_dataset['tld'].map(tld_weights).fillna(default_tld_weight)  # Default weight is 1 if tld is not in tld_weights

    if temporals == True:
        # Calculate 7-day and 30-day rolling average price and sales volume
        combined_dataset['7d_rolling_avg_price'] = combined_dataset['price_usd'].rolling(window=7).mean().fillna(0)
        combined_dataset['30d_rolling_avg_price'] = combined_dataset['price_usd'].rolling(window=30).mean().fillna(0)

        combined_dataset['7d_sales_volume'] = combined_dataset['price_usd'].rolling(window=7).sum().fillna(0)
        combined_dataset['30d_sales_volume'] = combined_dataset['price_usd'].rolling(window=30).sum().fillna(0)

        combined_dataset['cumulative_rolling_avg_price'] = combined_dataset['price_usd'].expanding().mean()

        combined_dataset['7d_domains_sold'] = combined_dataset['price_usd'].rolling(window=7).count().fillna(0)
        combined_dataset['30d_domains_sold'] = combined_dataset['price_usd'].rolling(window=30).count().fillna(0)
        combined_dataset['60d_domains_sold'] = combined_dataset['price_usd'].rolling(window=60).count().fillna(0)
        combined_dataset['90d_domains_sold'] = combined_dataset['price_usd'].rolling(window=90).count().fillna(0)

        combined_dataset['7d_rolling_std_dev'] = combined_dataset['price_usd'].rolling(window=7).std().fillna(0)
        combined_dataset['30d_rolling_std_dev'] = combined_dataset['price_usd'].rolling(window=30).std().fillna(0)

        combined_dataset['7d_rolling_median_price'] = combined_dataset['price_usd'].rolling(window=7).median().fillna(0)
        combined_dataset['30d_rolling_median_price'] = combined_dataset['price_usd'].rolling(window=30).median().fillna(0)

        combined_dataset['cumulative_sum_sales_volume'] = combined_dataset['price_usd'].expanding().sum().fillna(0)

        numeric_df = combined_dataset.select_dtypes(include=[float, int])

        correlation_matrix = numeric_df.corr(corr_type)

        target_corr = correlation_matrix[target].abs()
        print(f'threshold: {threshold}')

        print(f'target corr: {target_corr}')

        print(f'target corr under: {target_corr[target_corr < threshold]}')

        if threshold != None:

            columns_to_drop = target_corr[target_corr < threshold].index
            prophet_columns_to_drop = columns_to_drop.copy() + 'dt'
        else:

            columns_to_drop = ['90d_domains_sold', '60d_domains_sold', '30d_domains_sold', '7d_domains_sold', 'cumulative_sum_sales_volume',
                        'cumulative_rolling_avg_price','30d_rolling_median_price','7d_rolling_median_price',
                        '30d_rolling_std_dev']

            prophet_columns_to_drop = ['dt','90d_domains_sold', '60d_domains_sold', '30d_domains_sold', '7d_domains_sold', 'cumulative_sum_sales_volume',
                            'cumulative_rolling_avg_price','30d_rolling_median_price','7d_rolling_median_price',
                            '30d_rolling_std_dev']

    web3_data = combined_dataset.copy()

    
        
    # domain_data

    # %%
    # domain_data

    # %%
    

    combined_dataset = combined_dataset.drop_duplicates()
    combined_dataset['dt'] = pd.to_datetime(combined_dataset['dt'], errors='coerce')
    combined_dataset = combined_dataset.sort_values(by='dt')
    combined_dataset = combined_dataset.reset_index(drop=True)
    combined_dataset


    # %% [markdown]
    # # Feature Engineering

    # %%
    ## ETH Price

    prices_data

    # %%
   
    # %%
    combined_dataset

    features = combined_dataset.drop(columns=target).columns
    print(f'target:{target},\n features:{features}')

    # %%
    # %%
    combined_dataset

    # %%
 
    # 
    if temporals == True:
        # Calculate 7-day and 30-day rolling average price and sales volume
        combined_dataset['7d_rolling_avg_price'] = combined_dataset['price_usd'].rolling(window=7).mean().fillna(0)
        combined_dataset['30d_rolling_avg_price'] = combined_dataset['price_usd'].rolling(window=30).mean().fillna(0)

        combined_dataset['7d_sales_volume'] = combined_dataset['price_usd'].rolling(window=7).sum().fillna(0)
        combined_dataset['30d_sales_volume'] = combined_dataset['price_usd'].rolling(window=30).sum().fillna(0)

        combined_dataset['cumulative_rolling_avg_price'] = combined_dataset['price_usd'].expanding().mean()

        combined_dataset['7d_domains_sold'] = combined_dataset['price_usd'].rolling(window=7).count().fillna(0)
        combined_dataset['30d_domains_sold'] = combined_dataset['price_usd'].rolling(window=30).count().fillna(0)
        combined_dataset['60d_domains_sold'] = combined_dataset['price_usd'].rolling(window=60).count().fillna(0)
        combined_dataset['90d_domains_sold'] = combined_dataset['price_usd'].rolling(window=90).count().fillna(0)

        combined_dataset['7d_rolling_std_dev'] = combined_dataset['price_usd'].rolling(window=7).std().fillna(0)
        combined_dataset['30d_rolling_std_dev'] = combined_dataset['price_usd'].rolling(window=30).std().fillna(0)

        combined_dataset['7d_rolling_median_price'] = combined_dataset['price_usd'].rolling(window=7).median().fillna(0)
        combined_dataset['30d_rolling_median_price'] = combined_dataset['price_usd'].rolling(window=30).median().fillna(0)

        combined_dataset['cumulative_sum_sales_volume'] = combined_dataset['price_usd'].expanding().sum().fillna(0)

        numeric_df = combined_dataset.select_dtypes(include=[float, int])

        correlation_matrix = numeric_df.corr(corr_type)

        target_corr = correlation_matrix[target].abs()
        print(f'threshold: {threshold}')

        print(f'target corr: {target_corr}')

        print(f'target corr under: {target_corr[target_corr < threshold]}')

        if threshold != None:

            columns_to_drop = target_corr[target_corr < threshold].index
            prophet_columns_to_drop = columns_to_drop.copy() + 'dt'
        else:

            columns_to_drop = ['90d_domains_sold', '60d_domains_sold', '30d_domains_sold', '7d_domains_sold', 'cumulative_sum_sales_volume',
                        'cumulative_rolling_avg_price','30d_rolling_median_price','7d_rolling_median_price',
                        '30d_rolling_std_dev']

            prophet_columns_to_drop = ['dt','90d_domains_sold', '60d_domains_sold', '30d_domains_sold', '7d_domains_sold', 'cumulative_sum_sales_volume',
                            'cumulative_rolling_avg_price','30d_rolling_median_price','7d_rolling_median_price',
                            '30d_rolling_std_dev']
    else:
    # Print the resulting dataframe
        # print(combined_dataset[['dt', 'domain', 'price_usd', '7d_rolling_avg_price', '30d_rolling_avg_price', '7d_sales_volume', '30d_sales_volume','cumulative_rolling_avg_price']])

        

        # %%
        numeric_data = combined_dataset.select_dtypes(include=[float, int])

        # Calculate correlation of 'price_usd' with other numeric columns
        correlation_with_target = numeric_data.corr()[target]

        # Print the correlations
        print(correlation_with_target.sort_values())

        # %%
        columns_to_drop = []

        prophet_columns_to_drop = ['dt']

    print(f'cols to drop: {columns_to_drop}')
    print(f'prophet to drop: {prophet_columns_to_drop}')
    print(f'features: {features}')

    if columns_to_drop is None:
        columns_to_drop = []

    columns_to_drop = pd.Index(columns_to_drop)  # Ensure it's array-like
    # Drop columns from Index
    gen_features = features.difference(columns_to_drop)
    print(f'gen features')
    gen_features

    # %%
    prophet_features = features.difference(prophet_columns_to_drop)
    print(f'prophet features')
    prophet_features

    # %%
    X = combined_dataset[gen_features]
    y = combined_dataset[target]

    X_web3 = web3_data[gen_features]
    y_web3 = web3_data[target]
    
    return X, y, prophet_features, gen_features, target, combined_dataset, features, web3_data, X_web3, y_web3