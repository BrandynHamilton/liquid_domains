mints_query = """

SELECT
    DATE_TRUNC('HOUR', BLOCK_TIMESTAMP) AS day, tokenid, tx_hash
  FROM
    optimism.nft.ez_nft_transfers
  where
    nft_address = lower('0xBB7B805B257d7C76CA9435B3ffe780355E4C4B17')
    AND event_type = 'mint'
  order by
    block_timestamp desc
"""

sales_query = """

  SELECT
    DATE_TRUNC('HOUR', BLOCK_TIMESTAMP) AS day, tokenid, tx_hash, price
  FROM
    optimism.nft.ez_nft_sales
  WHERE
    NFT_ADDRESS = LOWER('0xBB7B805B257d7C76CA9435B3ffe780355E4C4B17')
    AND event_type = 'sale'
"""

eth_price_query = """

SELECT
  hour AS day,
  price
FROM
  ethereum.price.ez_prices_hourly
WHERE
  symbol = 'WETH'
  AND date_trunc('DAY', hour) >= date('2023-06-01')
order by
  hour asc
"""




