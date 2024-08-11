# %%
import requests
import pandas as pd

import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# %%
def extract_success_timestamp(data):
    items = data.get('items', [])
    records = []
    for item in items:
        success = item.get('success', None)
        timestamp = item.get('timestamp', None)
        records.append({'success': success, 'timestamp': timestamp})
    return records

def blockscout():
    base_url = "https://base-sepolia.blockscout.com/api/v2/addresses/0xa4e91145fd2370eca42186b9614ae3df398832cd/internal-transactions?filter=to%20%7C%20from"
    op_url = "https://optimism-sepolia.blockscout.com/api/v2/addresses/0x3e6f168587f9721a31f2fa1a560e6ab36d3b8c69/internal-transactions?filter=to%20%7C%20from"
    
    # Initialize records
    base_records = []
    op_records = []

    try:
        # Make the GET request for base
        base_response = requests.get(base_url)
        base_response.raise_for_status()  # Raise HTTPError for bad responses
        # Extract data from base response
        base_data = base_response.json()
        base_records = extract_success_timestamp(base_data)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Base Sepolia: {e}")

    try:
        # Make the GET request for op
        op_response = requests.get(op_url)
        op_response.raise_for_status()  # Raise HTTPError for bad responses
        # Extract data from op response
        op_data = op_response.json()
        op_records = extract_success_timestamp(op_data)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Optimism Sepolia: {e}")

    # Convert to DataFrames
    base_consumer = pd.DataFrame(base_records)
    op_consumer = pd.DataFrame(op_records)
    # %%
    base_consumer['call_success'] = base_consumer['success'].apply(lambda x: 1 if x else 0)
    base_consumer['call_fail'] = base_consumer['success'].apply(lambda x: 0 if x else 1)

    # Drop the original 'success' column if no longer needed
    base_consumer = base_consumer.drop(columns=['success'])

    print(base_consumer)

    # %%
    op_consumer['call_success'] = op_consumer['success'].apply(lambda x: 1 if x else 0)
    op_consumer['call_fail'] = op_consumer['success'].apply(lambda x: 0 if x else 1)

    # Drop the original 'success' column if no longer needed
    op_consumer = op_consumer.drop(columns=['success'])

    # %%
    base_consumer['timestamp'] = pd.to_datetime(base_consumer['timestamp'])

    # %%
    op_consumer['timestamp'] = pd.to_datetime(op_consumer['timestamp'])

    # %%
    base_consumer['hourly'] = base_consumer['timestamp'].dt.floor('H')

    # Step 2: Group by the hourly timestamp and sum the success counts
    hourly_aggregated = base_consumer.groupby('hourly')[['call_success','call_fail']].sum().reset_index()

    # Display the result
    print(hourly_aggregated)

    # %%
    op_consumer['hourly'] = op_consumer['timestamp'].dt.floor('H')

    # Step 2: Group by the hourly timestamp and sum the success counts
    op_hourly_aggregated = op_consumer.groupby('hourly')[['call_success','call_fail']].sum().reset_index()

    # Display the result
    print(op_hourly_aggregated)

    # %%
    # Assuming hourly_aggregated is the DataFrame we created earlier
    # Create a complete range of hourly timestamps
    full_range = pd.date_range(start=hourly_aggregated['hourly'].min(), 
                            end=hourly_aggregated['hourly'].max(), 
                            freq='H')

    # Convert to DataFrame
    full_range_df = pd.DataFrame(full_range, columns=['hourly'])

    # Merge the full range with the aggregated data
    hourly_filled = pd.merge(full_range_df, hourly_aggregated, on='hourly', how='left')

    # Fill missing success counts with 0
    hourly_filled['call_success'] = hourly_filled['call_success'].fillna(0)
    hourly_filled['call_fail'] = hourly_filled['call_fail'].fillna(0)


    # Display the result
    print(hourly_filled)


    # %%
 

    # Assuming hourly_aggregated is the DataFrame we created earlier
    # Create a complete range of hourly timestamps
    full_range = pd.date_range(start=op_hourly_aggregated['hourly'].min(), 
                            end=op_hourly_aggregated['hourly'].max(), 
                            freq='H')

    # Convert to DataFrame
    full_range_df = pd.DataFrame(full_range, columns=['hourly'])

    # Merge the full range with the aggregated data
    op_hourly_filled = pd.merge(full_range_df, op_hourly_aggregated, on='hourly', how='left')

    # Fill missing success counts with 0
    op_hourly_filled['call_success'] = op_hourly_filled['call_success'].fillna(0)
    op_hourly_filled['call_fail'] = op_hourly_filled['call_fail'].fillna(0)


    # Display the result
    print(op_hourly_filled)


    # %%
    base_sepolia_chart = make_subplots(specs=[[{"secondary_y": True}]])
            
    base_sepolia_chart.add_trace(
        go.Bar(
            x=hourly_filled['hourly'],
            y=hourly_filled['call_success'],
            name='Successful Calls',
        ),
        secondary_y=False
    )

    base_sepolia_chart.add_trace(
        go.Bar(
            x=hourly_filled['hourly'],
            y=hourly_filled['call_fail'],
            name='Failed Calls',
        ),
        secondary_y=False
    )

    base_sepolia_chart.update_layout(
        barmode='stack',
        legend=dict(yanchor="top", y=1.1, xanchor="left", x=0.01,orientation="h",bgcolor='rgba(0,0,0,0)'
    ))
    base_sepolia_chart.update_layout(
        title='Base-Sepolia Chainlink Calls per Hour',
        # barmode='group'  # Set the bar mode to either 'group' for side-by-side or 'stack' for stacked
    )

    # base_sepolia_chart.show()

    # %%
    op_sepolia_chart = make_subplots(specs=[[{"secondary_y": True}]])
            
    op_sepolia_chart.add_trace(
        go.Bar(
            x=op_hourly_filled['hourly'],
            y=op_hourly_filled['call_success'],
            name='Successful Calls',
        ),
        secondary_y=False
    )

    op_sepolia_chart.add_trace(
        go.Bar(
            x=op_hourly_filled['hourly'],
            y=op_hourly_filled['call_fail'],
            name='Failed Calls',
        ),
        secondary_y=False
    )

    op_sepolia_chart.update_layout(
        barmode='stack',
        legend=dict(yanchor="top", y=1.1, xanchor="left", x=0.01,orientation="h",bgcolor='rgba(0,0,0,0)'
    ))
    op_sepolia_chart.update_layout(
        title='Optimism-Sepolia Chainlink Calls per Hour',
        # barmode='group'  # Set the bar mode to either 'group' for side-by-side or 'stack' for stacked
    )

    # op_sepolia_chart.show()

    op_json = pio.to_json(op_sepolia_chart)
    base_json = pio.to_json(base_sepolia_chart)
    

    return op_json, base_json

# %%



