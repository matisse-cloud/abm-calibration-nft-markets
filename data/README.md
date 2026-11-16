# Data

The NFT dataset was provided for the CentraleSupélec ST4 course and is **not redistributed** in this repository.

To run notebooks 02 and 03, place `sales.csv` in this folder; notebook 02 converts it to `sales.parquet` on first run.

| | |
|---|---|
| Rows | 2,382,584 sales (`event_type = "sale"`) |
| Period | 2021-05-01 → 2025-04-30 |
| Wallets | buyer and seller addresses, anonymised as 16-hex-character hashes |
| Known issue | 92,873 rows (3.9%) have an `event_timestamp` that does not parse with `%Y-%m-%d %H:%M:%S` |

Main columns: `event_type`, `event_timestamp`, `nft_collection`, `nft_identifier` (token number within a collection), `nft_name`, `buyer`, `seller`, `quantity`, `payment_usd_price`, `log_usd_price`, `eth_usd_price`, `avg_gas_price`, `nft_updated_at` (metadata update time, *not* the trade time).
