import pandas as pd
import numpy as np
import random
import os
import sys
import requests
import time
import datetime as dt

from flipside import Flipside

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

def hourly(df):
    df['dt'] = df['dt'].dt.strftime('%Y-%m-%d %H-00-00')
    df['dt'] = pd.to_datetime(df['dt'])
    return df

def flipside_api_results(key, query):
  flipside = key
  query_result_set = flipside.query(query)
  # what page are we starting on?
  current_page_number = 1

  # How many records do we want to return in the page?
  page_size = 1000

  # set total pages to 1 higher than the `current_page_number` until
  # we receive the total pages from `get_query_results` given the 
  # provided `page_size` (total_pages is dynamically determined by the API 
  # based on the `page_size` you provide)

  total_pages = 2


  # we'll store all the page results in `all_rows`
  all_rows = []

  while current_page_number <= total_pages:
    results = flipside.get_query_results(
      query_result_set.query_id,
      page_number=current_page_number,
      page_size=page_size
    )

    total_pages = results.page.totalPages
    if results.records:
        all_rows = all_rows + results.records
    
    current_page_number += 1

  return pd.DataFrame(all_rows)

