unstoppable_domains = """
  SELECT
      DATE_TRUNC('HOUR', BLOCK_TIMESTAMP) AS day, tokenid, price
  FROM
    polygon.nft.ez_nft_sales
  WHERE
    NFT_ADDRESS = LOWER('0xa9a6A3626993D487d2Dbda3173cf58cA1a9D9e9f')
    AND event_type = 'sale'
"""

three_dns_sales = """

  SELECT
    DATE_TRUNC('HOUR', BLOCK_TIMESTAMP) AS day, tokenid, price
  FROM
    optimism.nft.ez_nft_sales
  WHERE
    NFT_ADDRESS = LOWER('0xBB7B805B257d7C76CA9435B3ffe780355E4C4B17')
    AND event_type = 'sale'
"""

optimism_name_service_sales = """

  SELECT
      DATE_TRUNC('HOUR', BLOCK_TIMESTAMP) AS day, tokenid, price
    FROM
      optimism.nft.ez_nft_sales
    WHERE
      NFT_ADDRESS = LOWER('0x4454Ee4F432f15e0D6479Cfe5954E08bf0a08B02')
      AND event_type = 'sale'


"""

Optimistic_Domains_Sales = """
  SELECT
      DATE_TRUNC('HOUR', BLOCK_TIMESTAMP) AS day, tokenid, price
    FROM
      optimism.nft.ez_nft_sales
    WHERE
      NFT_ADDRESS = LOWER('0xC16aCAdf99E4540E6f4E6Da816fd6D2A2C6E1d4F')
      AND event_type = 'sale'

"""

ENS_domain_sales = """

  SELECT
      DATE_TRUNC('HOUR', BLOCK_TIMESTAMP) AS day, tokenid, price
    FROM
      ethereum.nft.ez_nft_sales
    WHERE
      NFT_ADDRESS = LOWER('0xD4416b13d2b3a9aBae7AcD5D6C2BbDBE25686401')
      AND event_type = 'sale'
"""

base_domain_sales = """

  SELECT
      DATE_TRUNC('HOUR', BLOCK_TIMESTAMP) AS day, tokenid, price
    FROM
      base.nft.ez_nft_sales
    WHERE
      NFT_ADDRESS = LOWER('0x836198F984431EcdC97A7549C1Bd6B3Cd9E7a89B')
      AND event_type = 'sale'

"""

eth_price = """

  select
    hour as dt,
    symbol,
    price
  from
    ethereum.price.ez_prices_hourly
  where
    symbol in('WETH', 'MATIC')
    AND date_trunc('day', dt) >= '	2022-06-01'
  order by
    dt DESC


"""