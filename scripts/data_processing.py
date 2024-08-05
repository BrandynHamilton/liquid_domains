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
from models.forecasters import EnsemblePredictor, Prophet_Domain_Valuator, Domain_Valuator, train_ridge_model, train_randomforest_model, train_prophet_model
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

def process_data():

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
    
    return X, y, prophet_features, gen_features, target, combined_dataset, features