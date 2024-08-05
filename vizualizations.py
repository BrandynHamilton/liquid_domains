# %%
# %%
import pandas as pd
import numpy as np
import random
import os
import sys
import requests
import time
import datetime as dt

import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

# %%

def create_visualizations(combined_dataset):
    # %%
    combined_dataset.columns





    # %% [markdown]
    # # Add web2 vs web3 identifier by data source for marketplace

    # %%
    viz_data = combined_dataset.copy()
    print(f'viz data: {viz_data.head()} \n{viz_data.tail()}')

    # %%
    viz_data.columns

    # %%
    # viz_data.index = pd.to_datetime(viz_data.index)
    viz_data.set_index('dt', inplace=True)

    # %%
    daily_sales_vol = viz_data['price_usd'].resample('d').sum()
    daily_sales = viz_data['domain'].resample('d').count()


    print(f'{daily_sales, daily_sales_vol}')

    # %%
    daily_sales_aggregate = pd.DataFrame({
        'daily_sales_vol': daily_sales_vol,
        'daily_sales': daily_sales
    })

    # %%
    daily_sales_aggregate['7d_sales_volume'] = daily_sales_aggregate['daily_sales_vol'].rolling(window=7).sum()
    daily_sales_aggregate['30d_sales_volume'] = daily_sales_aggregate['daily_sales_vol'].rolling(window=30).sum()
    daily_sales_aggregate['7d_rolling_avg_price'] = daily_sales_aggregate['daily_sales_vol'].rolling(window=7).mean()
    daily_sales_aggregate['30d_rolling_avg_price'] = daily_sales_aggregate['daily_sales_vol'].rolling(window=30).mean()
    daily_sales_aggregate['7d_domains_sold'] = daily_sales_aggregate['daily_sales'].rolling(window=7).sum()
    daily_sales_aggregate['30d_domains_sold'] = daily_sales_aggregate['daily_sales'].rolling(window=30).sum()
    daily_sales_aggregate['7d_rolling_std_dev'] = daily_sales_aggregate['daily_sales'].rolling(window=7).std()
    daily_sales_aggregate['30d_rolling_std_dev'] = daily_sales_aggregate['daily_sales'].rolling(window=30).std()

    # Calculate cumulative metrics
    daily_sales_aggregate['cumulative_sum_sales_volume'] = daily_sales_aggregate['daily_sales_vol'].cumsum()
    daily_sales_aggregate['cumulative_rolling_avg_price'] = daily_sales_aggregate['daily_sales_vol'].expanding().mean()

    # %%
    temporals = ['7d_rolling_avg_price','30d_rolling_avg_price','7d_sales_volume','30d_sales_volume','cumulative_rolling_avg_price','7d_domains_sold','30d_domains_sold','7d_rolling_std_dev','30d_rolling_std_dev']
    cumulatives = ['cumulative_sum_sales_volume','cumulative_rolling_avg_price']

    # %%
    top_250 = daily_sales_aggregate.head(250)

    # %%
    cumulative_sales_chart = make_subplots(specs=[[{"secondary_y": True}]])
        
    cumulative_sales_chart.add_trace(
        go.Scatter(
            x=daily_sales_aggregate.index,
            y=daily_sales_aggregate['cumulative_sum_sales_volume'],
            name='cumulative_sum_sales_volume',
            stackgroup='one'
        ),
        secondary_y=False
    )
    cumulative_sales_chart.update_layout(
        title='Cumulative Sales Volume',
        # barmode='group'  # Set the bar mode to either 'group' for side-by-side or 'stack' for stacked
    )

    cumulative_sales_chart.update_xaxes(title_text="Date")

    # cumulative_sales_chart.show()

    # %%
    def sales_7_30(start=None, end=None):
        test_fig = make_subplots(specs=[[{"secondary_y": True}]])
            
        test_fig.add_trace(
            go.Scatter(
                x=daily_sales_aggregate.index,
                y=daily_sales_aggregate['7d_sales_volume'],
                name='7d_sales_volume',
                stackgroup='one'
            ),
            secondary_y=False
        )
        test_fig.add_trace(
            go.Scatter(
                x=daily_sales_aggregate.index,
                y=daily_sales_aggregate['30d_sales_volume'],
                name='30d_sales_volume',
                stackgroup='one'
            ),
            secondary_y=False
        )
        test_fig.update_layout(
            title='7d and 30d Sales Volume',
            # barmode='group'  # Set the bar mode to either 'group' for side-by-side or 'stack' for stacked
        )

        test_fig.update_xaxes(title_text="Date")

        return test_fig

    # %%
    ma_plot = sales_7_30()

    # %% [markdown]
    # def make_chart(df, columns, start=None, end=None):
    #     fig = make_subplots(specs=[[{"secondary_y": True}]])
    # 
    #     print('columns', columns)
    # 
    #     for col in df.columns:
    #         if col in columns:
    #             print('col:', col)
    #             fig.add_trace(
    #                 go.Scatter(
    #                     x=df.index,
    #                     y=df[col],
    #                     name=col,
    #                     mode='lines'
    #                 ),
    #                 secondary_y=False
    #             )
    # 
    #     fig.update_layout(
    #         title='Daily Sales Volume and Number of Sales'
    #     )
    # 
    #     fig.update_xaxes(title_text="Date")
    # 
    #     fig.show()
    # 
    #     return fig

    # %% [markdown]
    # def make_chart(df, columns, y2=False, y2_col=None, start=None, end=None):
    #     fig = make_subplots(specs=[[{"secondary_y": True}]])
    # 
    #     print('columns', columns)
    # 
    #     for col in df.columns:
    #         if col in columns:
    #             print('col:', col)
    #             if col != y2_col:
    #                 fig.add_trace(
    #                     go.Scatter(
    #                         x=df.index,
    #                         y=df[col],
    #                         name=col,
    #                         mode='lines',
    #                         stackgroup='one'
    #                     ),
    #                     secondary_y=False
    #                 )
    #             elif col == y2_col:
    #                 print('col:', col)
    #                 fig.add_trace(
    #                     go.Scatter(
    #                         x=df.index,
    #                         y=df[col],
    #                         name=col,
    #                         mode='lines',
    #                         stackgroup='one'
    #                     ),
    #                     secondary_y=True
    #                 )
    # 
    #     fig.update_layout(
    #         title='Daily Sales Volume and Number of Sales'
    #     )
    # 
    #     fig.update_xaxes(title_text="Date")
    # 
    #     fig.show()
    # 
    #     return fig

    # %% [markdown]
    # cumulative_sales_chart = make_subplots(specs=[[{"secondary_y": True}]])
    #     
    # cumulative_sales_chart.add_trace(
    #     go.Scatter(
    #         x=daily_sales_aggregate.index,
    #         y=daily_sales_aggregate['cumulative_rolling_avg_price'],
    #         name='cumulative_sum_sales_volume',
    #         stackgroup='one'
    #     ),
    #     secondary_y=False
    # )
    # cumulative_sales_chart.update_layout(
    #     title='Cumulative Rolling Avg Price',
    #     # barmode='group'  # Set the bar mode to either 'group' for side-by-side or 'stack' for stacked
    # )
    # 
    # cumulative_sales_chart.update_xaxes(title_text="Date")
    # 
    # cumulative_sales_chart.show()
    # 

    # %%

    sold_domains_fig = make_subplots(specs=[[{"secondary_y": True}]])
        
    sold_domains_fig.add_trace(
        go.Scatter(
            x=daily_sales_aggregate.index,
            y=daily_sales_aggregate['7d_domains_sold'],
            name='7d domains sold',
            stackgroup='one'
        ),
        secondary_y=False
    )
    sold_domains_fig.add_trace(
        go.Scatter(
            x=daily_sales_aggregate.index,
            y=daily_sales_aggregate['30d_domains_sold'],
            name='30d domains sold',
            stackgroup='one'
        ),
        secondary_y=False
    )
    sold_domains_fig.update_layout(
        title='7d and 30d # of Domains Sold',
        # barmode='group'  # Set the bar mode to either 'group' for side-by-side or 'stack' for stacked
    )

    sold_domains_fig.update_xaxes(title_text="Date")



    # %% [markdown]
    # 
    # 
    # test_fig = make_subplots(specs=[[{"secondary_y": True}]])
    #     
    # test_fig.add_trace(
    #     go.Scatter(
    #         x=daily_sales_aggregate.index,
    #         y=daily_sales_aggregate['7d_rolling_std_dev'],
    #         name='7d_sales_volume',
    #         stackgroup='one'
    #     ),
    #     secondary_y=False
    # )
    # test_fig.add_trace(
    #     go.Scatter(
    #         x=daily_sales_aggregate.index,
    #         y=daily_sales_aggregate['30d_rolling_std_dev'],
    #         name='30d_sales_volume',
    #         stackgroup='one'
    #     ),
    #     secondary_y=False
    # )
    # test_fig.update_layout(
    #     title='7d and 30d sales_volume',
    #     # barmode='group'  # Set the bar mode to either 'group' for side-by-side or 'stack' for stacked
    # )
    # 
    # test_fig.update_xaxes(title_text="Date")
    # 
    # 

    # %% [markdown]
    # test_fig = make_subplots(specs=[[{"secondary_y": True}]])
    #     
    # test_fig.add_trace(
    #     go.Scatter(
    #         x=daily_sales_aggregate.index,
    #         y=daily_sales_aggregate['daily_sales'],
    #         name='30d_sales_volume',
    #         stackgroup='one'
    #     ),
    #     secondary_y=True
    # )
    # test_fig.update_layout(
    #     title='7d and 30d sales_volume',
    #     
    #     # barmode='group'  # Set the bar mode to either 'group' for side-by-side or 'stack' for stacked
    # )
    # 
    # test_fig.update_xaxes(title_text="Date")
    # 
    # 

    # %% [markdown]
    # test_fig = make_subplots(specs=[[{"secondary_y": True}]])
    #     
    # test_fig.add_trace(
    #     go.Scatter(
    #         x=daily_sales_aggregate.index,
    #         y=daily_sales_aggregate['daily_sales_vol'],
    #         stackgroup='one'
    #     ),
    #     secondary_y=False
    # )
    # 
    # test_fig.update_layout(
    #     title='7d and 30d sales_volume',
    #     
    #     # barmode='group'  # Set the bar mode to either 'group' for side-by-side or 'stack' for stacked
    # )
    # 
    # test_fig.update_xaxes(title_text="Date")
    # 
    # 

    # %%

    rolling_avg_plot = make_subplots(specs=[[{"secondary_y": True}]])
        
    rolling_avg_plot.add_trace(
        go.Scatter(
            x=daily_sales_aggregate.index,
            y=daily_sales_aggregate['7d_rolling_avg_price'],
            name='7d rolling avg price',
            stackgroup='one'
        ),
        secondary_y=False
    )
    rolling_avg_plot.add_trace(
        go.Scatter(
            x=daily_sales_aggregate.index,
            y=daily_sales_aggregate['30d_rolling_avg_price'],
            name='30d rolling avg price',
            stackgroup='one'
        ),
        secondary_y=False
    )
    rolling_avg_plot.update_layout(
        title='7d and 30d rolling avg prices',
        # barmode='group'  # Set the bar mode to either 'group' for side-by-side or 'stack' for stacked
    )

    rolling_avg_plot.update_xaxes(title_text="Date")

    cumulative_sales_chart_json = pio.to_json(cumulative_sales_chart)
    ma_plot_json = pio.to_json(ma_plot)
    sold_domains_fig_json = pio.to_json(sold_domains_fig)
    rolling_avg_plot_json = pio.to_json(rolling_avg_plot)

    return cumulative_sales_chart, ma_plot, sold_domains_fig, rolling_avg_plot



    # %%



