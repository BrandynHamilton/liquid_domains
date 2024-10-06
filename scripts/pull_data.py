from scripts.apis import *
from sql_queries.sql import eth_price, Optimistic_Domains_Sales as Optimistic_Domains_Sales_query

flipside_api_key = os.getenv('FLIPSIDE_API_KEY')
alchemy_api_key = os.getenv('ALCHEMY_API_KEY')
opensea_api_key = os.getenv('OPENSEA_API_KEY')

def pull_data(api=False):
        optimistic_domains_sales_path = '../data/optimistic_domains_sales.csv'
        ens_sales_path = '../data/ens_metadata.json'
        Optimistic_domains_path = '../data/optimistic_domains_metadata.json'
        three_dns_path = '../data/3dns_metadata.json'
        base_domains_path = '../data/base_metadata.json'
        unstoppable_sales_path = '../data/unstoppable_metadata.json'
        optimism_name_service_path = '../data/optimism_name_service_metadata.json'
        prices_path = '../data/prices.csv'

        if api == True:
            print('Pulling Fresh Data...')
            # Pull fresh data from API
            ens_data = fetch_all_events(api_key=opensea_api_key,collection='ens')
            ens_data.to_json(ens_sales_path, orient='records', date_format='iso')

            optimism_name_service_data = fetch_all_events(api_key=opensea_api_key,collection='optimism-name-service')
            optimism_name_service_data.to_json(optimism_name_service_path, orient='records')
            
            unstoppable_sales_data = fetch_all_events(api_key=opensea_api_key,collection='unstoppable-domains')
            unstoppable_sales_data.to_json(unstoppable_sales_path, orient='records', date_format='iso')
            
            base_domains_metadata = fetch_all_events(api_key=opensea_api_key,collection='basedomainnames')
            base_domains_metadata.to_json(base_domains_path, orient='records')
            
            Three_DNS_data = fetch_all_events(api_key=opensea_api_key,collection='3dns-powered-domains')
            Three_DNS_data.to_json(three_dns_path, orient='records')
            
            optimistic_domains_sales = flipside_api_results(Optimistic_Domains_Sales_query,flipside_api_key)
            optimistic_domains_sales.to_csv(optimistic_domains_sales_path, index=False)

            prices = flipside_api_results(eth_price, flipside_api_key)
            prices.to_csv(prices_path,index=False)

            optimistic_domains = alchemy_metadata_api(alchemy_api_key, 'optimism', '0xC16aCAdf99E4540E6f4E6Da816fd6D2A2C6E1d4F')
            optimistic_domains.to_json(Optimistic_domains_path, orient='records')

        else:
            print('Loading Existing Data...')

            ens_data = pd.read_json(ens_sales_path, orient='records')
            optimistic_domains = pd.read_json(Optimistic_domains_path, orient='records')
            Three_DNS_data = pd.read_json(three_dns_path, orient='records')
            base_domains_metadata_pd = pd.read_json(base_domains_path, orient='records')
            optimism_name_service_data = pd.read_json(optimism_name_service_path, orient='records')
            unstoppable_sales_data = pd.read_json(unstoppable_sales_path, orient='records')
            optimistic_domains_sales = pd.read_csv(optimistic_domains_sales_path)
            prices = pd.read_csv(prices_path)


        # DomainData = namedtuple('DomainData', ['optimistic_domains_sales', 'ens_data', 'optimistic_domains', 'Three_DNS_data', 'base_domains_metadata_pd','unstoppable_sales_data'])

        # data = DomainData(
        #     optimistic_domains_sales=optimistic_domains_sales,
        #     ens_data=ens_data,
        #     optimistic_domains=optimistic_domains,
        #     Three_DNS_data=Three_DNS_data,
        #     base_domains_metadata_pd=base_domains_metadata_pd,
        #     unstoppable_sales_data=unstoppable_sales_data
        

        data = {
            'ens': ens_data,
            'optimistic_domains': optimistic_domains,
            'Three_DNS_data': Three_DNS_data,
            'base_domains_metadata_pd': base_domains_metadata_pd,
            'unstoppable_sales_data': unstoppable_sales_data,
            "optimistic_domains_sales": optimistic_domains_sales,
            "optimism_name_service_data": optimism_name_service_data,
            'prices_data':prices
        }

        for name, df in data.items():
            print(f"--- {name} ---")
            print("Head:")
            print(df.head())
            print("\nTail:")
            print(df.tail())
            print("\n" + "="*40 + "\n")  # Just for formatting

        return data

# if __name__ == '__main__':
#     data = pull_data(api=False)
#     print(f'data retrieved')