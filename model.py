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
# from sql_queries.sql_scripts import three_dns_sales

# %%
pd.options.display.float_format = '{:,.2f}'.format
three_dns_sales = """

  SELECT
    DATE_TRUNC('HOUR', BLOCK_TIMESTAMP) AS day, tokenid, price
  FROM
    optimism.nft.ez_nft_sales
  WHERE
    NFT_ADDRESS = LOWER('0xBB7B805B257d7C76CA9435B3ffe780355E4C4B17')
    AND event_type = 'sale'
"""

# %%
current_directory = os.getcwd()
current_directory

# %%
load_dotenv()

# %%
seed = 20
set_random_seed(seed)

# %%
flipside_api_key = os.getenv('FLIPSIDE_API_KEY')
alchemy_api_key = os.getenv('ALCHEMY_API_KEY')
opensea_api_key = os.getenv('OPENSEA_API_KEY')

print(alchemy_api_key)

# %%
def alchemy_metadata_api(api_key, network, contract_address):
    if network == 'optimism':
        network = 'opt'
    elif network == 'ethereum':
        network = 'eth'
    elif network == 'base':
        network = 'base'
    # Replace with your actual API key
    base_url = f"https://{network}-mainnet.g.alchemy.com/nft/v3/{api_key}/getNFTsForContract"
    print(f'Base URL: {base_url}')
    headers = {"accept": "application/json"}

    # Pagination parameters
    page_key = None  # Initial key for pagination
    limit = 100  # Set the limit for the number of NFTs per request
    api_data = []  # To store all NFTs

    while True:
        params = {
            "contractAddress": contract_address,
            "withMetadata": "true",
            "limit": limit
        }
        
        if page_key:
            params["pageKey"] = page_key

        response = requests.get(base_url, headers=headers, params=params)
        data = response.json()
        
        if "nfts" in data:
            api_data.extend(data["nfts"])
            # print(data["nfts"])
        
        # Check if there's a next page key for pagination
        page_key = data.get("pageKey", None)
        
        if page_key is None:
            break

        print(f'Number added: {len(data["nfts"])} | Total number of NFTs: {len(api_data)}, Next page key: {page_key}')

    # Now `api_data` contains all the NFTs retrieved from the paginated API calls
    print(f"Total NFTs retrieved: {len(api_data)}")

    # Function to get metadata from tokenUri
    def fetch_metadata(token_uri):
        try:
            response = requests.get(token_uri)
            metadata = response.json()
            return metadata
        except:
            return {'name': 'No name available'}

    # Extract tokenId, name, and tokenUri from each NFT
    nft_info = []
    for nft in api_data:
        token_id = nft.get('tokenId', 'No token ID available')
        token_name = nft.get('name', 'No token ID available')
        
        
        nft_info.append({'tokenId': token_id, 'name': token_name})
    
    # Create DataFrame
    df = pd.DataFrame(nft_info)
    
    return df

# %% [markdown]
# optimism_name_service_metadata = alchemy_metadata_api(alchemy_api_key, 'optimism', '0xC16aCAdf99E4540E6f4E6Da816fd6D2A2C6E1d4F')

# %% [markdown]
# Three_DNS_metadata = alchemy_metadata_api(alchemy_api_key, 'optimism', '0xBB7B805B257d7C76CA9435B3ffe780355E4C4B17')

# %% [markdown]
# optimistic_domains = alchemy_metadata_api(alchemy_api_key, 'optimism', '0xC16aCAdf99E4540E6f4E6Da816fd6D2A2C6E1d4F')

# %%
Optimistic_domains_path = 'data/optimistic_domains_metadata.json'
# optimistic_domains.to_json(Optimistic_domains_path, orient='records')
optimistic_domains = pd.read_json(Optimistic_domains_path, orient='records')
# optimistic_domains.drop(columns=['tokenUri'], inplace=True)
optimistic_domains

# %%
domain_path = 'data/domain-name-sales.tsv'  
domain_data = pd.read_csv(domain_path, delimiter='\t')

# %%
domain_data.set_index('date', inplace=True)
domain_data = domain_data.drop(columns=['venue'])
domain_data.sort_index(inplace=True)
domain_data

# %%
def fetch_event_type(api_key, collection, event_type, all_events, params, headers):
    base_url = f"https://api.opensea.io/api/v2/events/collection/{collection}"
    params['event_type'] = event_type
    
    # Load the last timestamp/identifier
    
    page_count = 0
    while True:
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            fetched_events = data.get("asset_events", [])
            all_events.extend(fetched_events)
            
            if fetched_events:
                # Update the last timestamp/identifier to the latest one fetched
                last_event_time = fetched_events[-1].get("created_date")
            
            page_count += 1
            next_cursor = data.get("next")
            print(f"Fetching {event_type}: Page {page_count}, Events Fetched: {len(fetched_events)}, Total Events: {len(all_events)}, next cursor: {next_cursor}")
            
            if next_cursor:
                params['next'] = next_cursor
            else:
                break  # No more pages to fetch

            time.sleep(1)  # Delay between requests
        else:
            print(f"Failed to fetch {event_type} data: HTTP {response.status_code}, Response: {response.text}")
            break

def clean_data(domain_df):
    domain_df['nft_identifier'] = domain_df['nft'].apply(lambda x: x.get('identifier', 'No identifier available') if x else 'No identifier available')
    domain_df['nft_name'] = domain_df['nft'].apply(lambda x: x.get('name', 'No name available') if x else 'No name available')
    domain_df['token_amt_raw'] = domain_df['payment'].apply(lambda x: x.get('quantity', 'No name available') if x else 'No name available')
    domain_df['token_symbol'] = domain_df['payment'].apply(lambda x: x.get('symbol', 'No name available') if x else 'No name available')
    domain_df['token_decimals'] = domain_df['payment'].apply(lambda x: x.get('decimals', 'No name available') if x else 'No name available')
    domain_df['dt'] = pd.to_datetime(domain_df['event_timestamp'], unit='s')

    def wei_to_ether(quantity, decimals):
        try:
            return int(quantity) / (10 ** decimals)
        except ValueError:
            return None

    domain_df['token_amt_clean'] = domain_df.apply(lambda row: wei_to_ether(row['token_amt_raw'], row['token_decimals']) if row['token_amt_raw'] != 'No name available' and row['token_decimals'] != 'No name available' else None, axis=1)
    domain_df.dropna(inplace=True)
    return domain_df

# Display the updated DataFrame




def fetch_all_events(api_key, collection):
    headers = {
        "accept": "application/json",
        "x-api-key": api_key
    }
    params = {
        "limit": 50  # Adjust the limit as needed
    }

    all_events = []

    # Fetch listings
    # fetch_event_type(api_key, collection, "listing", all_events, params.copy(), headers)

    # Fetch sales
    fetch_event_type(api_key, collection, "sale", all_events, params.copy(), headers)

    # Save the fetched events to a DataFrame
    print(f"Total events fetched: {len(all_events)}")
    df = pd.DataFrame(all_events)
    clean_df = clean_data(df)
    return clean_df 





# %% [markdown]
# optimism_name_service_data = fetch_all_events(api_key=opensea_api_key,collection='optimism-name-service')
# 

# %%
optimism_name_service_path = 'data/optimism_name_service_metadata.json'
# optimism_name_service_data.to_json(optimism_name_service_path, orient='records')
optimism_name_service_data = pd.read_json(optimism_name_service_path, orient='records')
optimism_name_service_data = optimism_name_service_data[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]
optimism_name_service_data


# %% [markdown]
# Three_DNS_data = fetch_all_events(api_key=opensea_api_key,collection='3dns-powered-domains')
# 

# %%
three_dns_path = 'data/3dns_metadata.json'
# Three_DNS_data.to_json(three_dns_path, orient='records')
Three_DNS_data = pd.read_json(three_dns_path, orient='records')
# Three_DNS_data.dropna(inplace=True)
Three_DNS_data = Three_DNS_data[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]
Three_DNS_data

# %% [markdown]
# ens_sales_data = fetch_all_events(api_key=opensea_api_key,collection='ens')
# 

# %%
ens_sales_path = 'data/ens_metadata.json'
# ens_sales_data.to_json('data/ens_metadata.json', orient='records', date_format='iso')
ens_data = pd.read_json(ens_sales_path, orient='records')

# %%
ens_data = ens_data[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]

# %% [markdown]
# unstoppable_sales_data = fetch_all_events(api_key=opensea_api_key,collection='unstoppable-domains')
# 

# %%
unstoppable_sales_path = 'data/unstoppable_metadata.json'
# unstoppable_sales_data.to_json(unstoppable_sales_path, orient='records', date_format='iso')
unstoppable_sales_data = pd.read_json(unstoppable_sales_path, orient='records')
unstoppable_sales_data = unstoppable_sales_data[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]
unstoppable_sales_data

# %%
# unstoppable_sales_data['nft_identifier'] = unstoppable_sales_data['nft'].apply(lambda x: x.get('identifier', 'No identifier available') if x else 'No identifier available')
# unstoppable_sales_data['nft_name'] = unstoppable_sales_data['nft'].apply(lambda x: x.get('name', 'No name available') if x else 'No name available')
# unstoppable_sales_data.dropna(inplace=True)
# # Now you can view the DataFrame with the new columns
# print(unstoppable_sales_data[['event_type', 'closing_date', 'nft_identifier', 'nft_name']])
# unstoppable_sales_data = unstoppable_sales_data[['nft_identifier', 'nft_name']]

# %% [markdown]
# base_domains_metadata = fetch_all_events(api_key=opensea_api_key,collection='basedomainnames')

# %%
base_domains_path = 'data/base_metadata.json'
# base_domains_metadata.to_json(base_domains_path, orient='records')
base_domains_metadata_pd = pd.read_json(base_domains_path, orient='records')
base_domains_metadata_pd = base_domains_metadata_pd[['dt','token_symbol','token_amt_clean','nft_identifier','nft_name']]
base_domains_metadata_pd

# %% [markdown]
# 
# # Now you can view the DataFrame with the new columns
# print(base_domains_metadata_pd[['dt','event_type', 'closing_date', 'nft_identifier', 'nft_name','token_amt_clean','token_symbol']])
# base_domains_metadata_pd = base_domains_metadata_pd[['nft_identifier', 'nft_name','token_amt_clean','token_symbol']]

# %%
domain_data

# %% [markdown]
# combined_metadata = pd.concat([
#     base_domains_metadata_pd.dropna(),
#     unstoppable_sales_data.dropna(),
#     ens_data.dropna(),
#     Optimistic_domains_metadata_pd.dropna(),
#     Three_DNS_metadata_pd.dropna(),
#     optimism_name_service_metadata_pd.dropna()
# ], ignore_index=True)

# %% [markdown]
# combined_metadata.rename(columns={"tokenId":"TOKENID"}, inplace=True)

# %% [markdown]
# combined_metadata['TOKENID'].describe()

# %% [markdown]
# # Sales

# %%
ens_sales = pd.read_csv('data/ens_domain_sales.csv')
optimistic_domains_sales = pd.read_csv('data/optimistic_domains_sales.csv')
optimism_domain_service_sales = pd.read_csv('data/optimism_name_service_sales.csv')
base_domains_sales = pd.read_csv('data/base_domain_names_sales.csv')
unstoppable_domains_sales = pd.read_csv('data/unstoppable_domains_sales.csv')
three_dns_sales_data = pd.read_csv('data/three_dns_sales.csv')
prices_data = pd.read_csv('data/prices.csv')

# %%
prices_data = prices_data.dropna()
prices_data['SYMBOL'] = prices_data['SYMBOL'].replace('WETH', 'ETH')


prices_data = prices_data.pivot(index='DT',columns='SYMBOL',values='PRICE')
prices_data = prices_data.reset_index()
prices_data

# %%


# %%
combined_sales = pd.concat([
    ens_sales.dropna(),
    optimistic_domains_sales.dropna(),
    optimism_domain_service_sales.dropna(),
    base_domains_sales.dropna(),
    unstoppable_domains_sales.dropna(),
    three_dns_sales_data.dropna()
], ignore_index=True)

# %%
combined_sales = combined_sales.drop_duplicates()
combined_sales['DAY'] = pd.to_datetime(combined_sales['DAY'], errors='coerce')
combined_sales = combined_sales.sort_values(by='DAY')
combined_sales = combined_sales.reset_index(drop=True)
combined_sales


# %% [markdown]
# # Full Data Set and Feature Engineering

# %%
optimistic_domains_sales

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
domain_data = domain_data.reset_index()
domain_data = domain_data.rename(columns={"date":"dt","price":"price_usd"})
domain_data['dt'] = pd.to_datetime(domain_data['dt'])
domain_data['dt'] = domain_data['dt'].dt.tz_localize('UTC')
domain_data['dt'] = pd.to_datetime(domain_data['dt'])
domain_data

# %%
domain_data

# %%
combined_dataset = combined_dataset.rename(columns={'nft_name':'domain'})
combined_dataset = pd.concat([combined_dataset, domain_data], ignore_index=True)
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
combined_dataset = combined_dataset.drop(columns=['token_amt_clean'])

# %%
combined_dataset

# %%
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

# Print the resulting dataframe
print(combined_dataset[['dt', 'domain', 'price_usd', '7d_rolling_avg_price', '30d_rolling_avg_price', '7d_sales_volume', '30d_sales_volume','cumulative_rolling_avg_price']])

# %%
combined_dataset

# %%
combined_dataset['domain_length'] = combined_dataset['domain'].apply(len)
combined_dataset['num_vowels'] = combined_dataset['domain'].apply(lambda x: sum([1 for char in x if char in 'aeiou']))
combined_dataset['num_consonants'] = combined_dataset['domain'].apply(lambda x: sum([1 for char in x if char.isalpha() and char not in 'aeiou']))
combined_dataset['tld'] = combined_dataset['domain'].apply(lambda x: x.split('.')[-1])  # Extract TLD


# %% [markdown]
# tld_weights = {
#     'com': 1000000000,
#     'net': 8,
#     'org': 7,
#     'box': 3,
#     'eth': 2
#     # Add more TLDs and their weights as needed
# }
# 
# default_tld_weight = 1
# 
# combined_dataset['tld_weight'] = combined_dataset['tld'].map(tld_weights).fillna(default_tld_weight)  # Default weight is 1 if tld is not in tld_weights
# 

# %%
target = 'price_usd'
features = combined_dataset.drop(columns=target).columns
print(f'target:{target},\n features:{features}')

# %%
numeric_data = combined_dataset.select_dtypes(include=[float, int])

# Calculate correlation of 'price_usd' with other numeric columns
correlation_with_target = numeric_data.corr()[target]

# Print the correlations
print(correlation_with_target.sort_values())

# %%
columns_to_drop = ['90d_domains_sold', '60d_domains_sold', '30d_domains_sold', '7d_domains_sold', 'cumulative_sum_sales_volume',
                   'cumulative_rolling_avg_price','30d_rolling_median_price','7d_rolling_median_price',
                   '30d_rolling_std_dev']

prophet_columns_to_drop = ['dt','90d_domains_sold', '60d_domains_sold', '30d_domains_sold', '7d_domains_sold', 'cumulative_sum_sales_volume',
                   'cumulative_rolling_avg_price','30d_rolling_median_price','7d_rolling_median_price',
                   '30d_rolling_std_dev']

# Drop columns from Index
gen_features = features.difference(columns_to_drop)
gen_features

# %%
prophet_features = features.difference(prophet_columns_to_drop)
prophet_features

# %%
X = combined_dataset[gen_features]
y = combined_dataset[target]

# %% [markdown]
# ## Ridge Regression

# %% [markdown]
# # Preprocess categorical data (TLD) and handle missing values
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='median')),
#             ('scaler', StandardScaler())
#         ]), ['domain_length', 'num_vowels', 'num_consonants']),
#         ('cat', Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='most_frequent')),
#             ('onehot', OneHotEncoder(handle_unknown='ignore'))
#         ]), ['tld'])
#     ]
# )
# 
# # Create a pipeline with Ridge regression
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', Ridge())
# ])
# 
# # Define the parameter grid for hyperparameter tuning
# param_grid = {
#     'regressor__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]
# }

# %% [markdown]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
# 
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_root_mean_squared_error')
# grid_search.fit(X_train, y_train)
# 
# # Best model from grid search
# best_model = grid_search.best_estimator_
# 
# # Predict and evaluate
# y_pred = best_model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# 
# print(f'Best Alpha: {grid_search.best_params_["regressor__alpha"]}')
# print(f'MAE: {mae}')
# print(f'MSE: {mse}')
# print(f'R²: {r2}')

# %% [markdown]
# ## Prophet 

# %% [markdown]
# df_prophet = combined_dataset.copy()
# df_prophet.rename(columns={"dt":"ds","price_usd":"y"}, inplace=True)    
# df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
# 
# target = 'y'
# features = df_prophet[features].select_dtypes(include=[np.number]).columns
# 
# train_df, test_df = train_test_split(df_prophet, test_size=0.2, shuffle=False, random_state=42)
# 
# 
# model = Prophet()
# 
# for feature in features:
#     model.add_regressor(feature)
# 
# model.fit(train_df)
# 
# future = test_df[['ds']].copy()
# for feature in features:
#     # Use historical values for features from the training set
#     future[feature] = test_df[feature].values
# 
# forecast = model.predict(future)
# 
# y_true = test_df['y'].values
# y_pred = forecast['yhat'].values
# 
# r2 = r2_score(y_true, y_pred)
# mae = mean_absolute_error(y_true, y_pred)
# rmse = np.sqrt(mean_squared_error(y_true, y_pred))
# 
# print(f"R²: {r2}")
# print(f"MAE: {mae}")
# print(f"RMSE: {rmse}")

# %%
def train_ridge_model(X, y):

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
def train_randomforest_model(X, y):

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
def train_prophet_model(features):
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

# %% [markdown]
# # Valuation Model

# %%
from sklearn.ensemble import RandomForestRegressor

# %%
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

# %%
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

# %%
prophet_model, prophet_features = train_prophet_model(prophet_features)

# %%
ridge_model, ridge_features = train_ridge_model(X, y)

# %%
randomforest_model, random_forest_features = train_randomforest_model(X, y)

# %%
from sklearn.ensemble import VotingRegressor

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

# %%
ensemble_predictor = EnsemblePredictor(prophet_model, randomforest_model, ridge_model, prophet_features)

# Prepare data for prediction (X should be in the correct format for Prophet)
X = combined_dataset.copy()  # Ensure this DataFrame is correctly formatted
X['dt'] = X['dt'].dt.tz_localize(None)
X.rename(columns={'dt':'ds'}, inplace=True)

# Predict using the ensemble
ensemble_predictions = ensemble_predictor.predict(X, combined_dataset)

# %%
y_true = combined_dataset['price_usd'].values
r2 = r2_score(y_true, ensemble_predictions)
mae = mean_absolute_error(y_true, ensemble_predictions)
rmse = np.sqrt(mean_squared_error(y_true, ensemble_predictions))

print(f"Ensemble R²: {r2}")
print(f"Ensemble MAE: {mae}")
print(f"Ensemble RMSE: {rmse}")

# %%
domain = 'env.eth'

# %%
# Ensure features_data is up to date and correctly formatted
prophet_features_data = combined_dataset.copy()
prophet_features_data.rename(columns={"dt": "ds", "price_usd": "y"}, inplace=True)
# prophet_features_data['ds'] = pd.to_datetime(features_data['ds']).dt.tz_localize(None)  # Remove timezone if present

# Initialize the Domain_Valuator
prophet_valuator = Prophet_Domain_Valuator(domain, prophet_features, prophet_features_data)

# Prepare the model and get the domain value
prophet_valuator.model_prep()
prophet_domain_value = prophet_valuator.value_domain(prophet_model)

# %%
# Right now doesnt work w/ .com
features_data = combined_dataset.copy()
features_data['dt'] = features_data['dt'].dt.tz_localize(None)
features_data = features_data[features] 
ridge_valuator = Domain_Valuator(domain, X, y, gen_features, features_data, seed)

# Prepare the model and get the domain value
ridge_valuator.model_prep()
ridge_domain_value = ridge_valuator.value_domain(ridge_model)

# %%
# Right now doesnt work w/ .com
features_data = combined_dataset.copy()
features_data['dt'] = features_data['dt'].dt.tz_localize(None)
features_data = features_data[features] 
randomforest_valuator = Domain_Valuator(domain, X, y, gen_features, features_data, seed)

# Prepare the model and get the domain value
randomforest_valuator.model_prep()
randomforest_domain_value = randomforest_valuator.value_domain(randomforest_model)

# %%
ensemble_predictions = ensemble_predictor.predict(X, combined_dataset)

print(f'domain: {domain} \nprophet valuation: ${prophet_domain_value} \nrandom forest valuation: ${randomforest_domain_value} \nridge valuation: ${ridge_domain_value} \nensamble valuation: ${ensemble_predictions}')



# %%



