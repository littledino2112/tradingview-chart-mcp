# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an MCP (Model Context Protocol) server that captures TradingView chart images using Selenium browser automation. It provides tools for fetching chart snapshots with ticker symbols and timeframe intervals.

## Commands

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Running the Server
```bash
# Production (optimized with browser pooling)
python main_optimized.py

# Legacy (simple single-browser)
python main.py

# HTTP transport with custom port
python main_optimized.py --transport streamable-http --port 8003

# Disable browser pooling
python main_optimized.py --disable-pooling
```

### Running Tests
```bash
# Sequential performance test
python tests/test_mcp_agent_style.py --runs 5 --ticker BYBIT:BTCUSDT.P --interval 240

# Concurrent performance test
python tests/test_mcp_agent_style.py --concurrent 3 --ticker BYBIT:BTCUSDT.P --interval 240

# Test timeframes
python tests/test_timeframes.py --headless --use-save-shortcut
```

## Architecture

### Entry Points
- `main_optimized.py` - Production server with browser pooling for concurrent performance (70-80% faster)
- `main.py` - Legacy single-browser server for debugging/fallback

### Core Components
- `tview_scraper.py` - `TradingViewScraper` class that handles Selenium WebDriver, authentication, chart navigation, and clipboard-based image capture
- Both servers expose MCP tools via FastMCP framework

### Key Classes
- `TradingViewScraper` - Context manager for browser automation, handles session cookies and clipboard operations
- `OptimizedTradingViewMCPServer` (in main_optimized.py) - Manages browser pool with thread-safe access and async semaphore for concurrent requests

### MCP Tools Exposed
- `get_tradingview_chart_image(ticker, interval)` - Returns chart image (base64 data URL or S3 URL)
- `get_performance_stats()` - Returns performance metrics (optimized server only)

### Image Capture Methods
- **Save shortcut** (default, `MCP_SCRAPER_USE_SAVE_SHORTCUT=True`): Uses `Shift+Ctrl/Cmd+S` to capture directly to clipboard as base64
- **Traditional**: Uses `Alt+S` to get screenshot link, then converts to S3 image URL

### Valid Intervals
`'1'`, `'5'`, `'15'`, `'30'`, `'60'`, `'240'`, `'D'`, `'W'`, `'M'`

## Configuration

Required environment variables (set in `.env` or MCP client config):
- `TRADINGVIEW_SESSION_ID` - Session cookie from TradingView
- `TRADINGVIEW_SESSION_ID_SIGN` - Session signature cookie

Optional:
- `MCP_SCRAPER_HEADLESS` - `True`/`False` (default: `True`)
- `MCP_SCRAPER_WINDOW_WIDTH` / `MCP_SCRAPER_WINDOW_HEIGHT` - Browser dimensions (default: 1400)
- `MCP_SCRAPER_USE_SAVE_SHORTCUT` - Use clipboard image capture (default: `True`)
- `MCP_SCRAPER_CHART_PAGE_ID` - Custom TradingView chart layout ID

## Dependencies

- `mcp[cli]` - MCP server framework (FastMCP)
- `selenium` - Browser automation
- `python-dotenv` - Environment variable loading
