# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a comprehensive Bitcoin trading system with AI-powered signals using LSTM neural networks. The system features real-time price updates via WebSocket, paper trading capabilities, and a modern web interface built with FastAPI (backend) and Streamlit (frontend).

## Development Guidelines

- Do not use emojis in our developed products
- Always be succinct and precise in our documentation
- Always store documentation in ./docs folder

## Database Architecture

The system uses three SQLite databases:

1. **api_cache.db** - Caches API responses to reduce external calls
   - Tables: api_cache, cache_invalidation_log, cache_stats
   
2. **trading_system.db** - Core trading data and analytics
   - Tables: trades, positions, trading_limits, model_signals, backtest_results, signal_performance, feature_importance, market_regime, paper_portfolio, paper_trades, paper_performance
   - Key relationships: trades/positions via lot_id, paper_trades/paper_portfolio via portfolio_id
   
3. **historical_data.db** - Historical market data for analysis
   - Tables: ohlcv_data, onchain_data, sentiment_data

For detailed database schema and ERD, see: ./docs/database-erd.md