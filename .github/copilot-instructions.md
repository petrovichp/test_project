# GitHub Copilot Instructions for Crypto Trading ML Project

## Project Overview
This project is a research-first machine learning system for crypto trading on OKX, focusing on BTC, ETH, and SOL with 1-minute bars. The goal is to validate trading signals using an honest out-of-sample methodology before any production deployment.

## Key Components
- **Data Loading**: Use `data/loader.py` to load preprocessed data from the `cache/` directory. Avoid raw CSVs; always work with Parquet files for efficiency.
- **Feature Engineering**: Features are organized in the `features/` directory, with modules for orderbook, price, volume, and market data. Each module contains functions to generate specific features used in modeling.
- **Modeling**: Models are implemented in the `models/` directory, including volatility and direction prediction models. Use LightGBM for initial models before exploring deep learning approaches.

## Developer Workflows
- **Loading Data**: Use `load_meta(ticker)` to load metadata or `load(ticker, include_ob=True)` to load order book data. Ensure the cache is checked before recomputing data.
- **Running Models**: Execute models directly using `python3 -m models.volatility [ticker]` for volatility research. Follow the evaluation protocol outlined in `RESEARCH_PROMPT.md` for consistent results.
- **Testing and Validation**: Always validate models on a separate validation set before testing. Use the specified metrics (AUC, Spearman) to assess performance.

## Project Conventions
- **No Leakage**: Ensure that all features use only historical data up to the current time point. Avoid using future data in training.
- **No Random Splits**: Time-series data should never be shuffled. Use sequential or walk-forward validation methods only.
- **Caching**: Save all expensive intermediate results in the `cache/` directory as Parquet files. Always check the cache before recomputing.

## Integration Points
- **Data Sources**: The project relies on external CSV data sources located in `/Users/petrpogoraev/Documents/Projects/options_trading/DATA/last_source_data/`. Ensure that the data is structured correctly for loading.
- **Cross-Component Communication**: Features generated in the `features/` directory are used as inputs for models in the `models/` directory. Ensure that feature generation aligns with model expectations.

## Examples
- To load BTC metadata: `load_meta('btc')`
- To load BTC data with order book: `load('btc', include_ob=True)`
- To run volatility model: `python3 -m models.volatility btc`

## Conclusion
This document serves as a guide for AI coding agents to navigate the project effectively. Follow the outlined conventions and workflows to ensure consistency and efficiency in development.