[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/ertugrul59-tradingview-chart-mcp-badge.png)](https://mseep.ai/app/ertugrul59-tradingview-chart-mcp)

# MCP Server - TradingView Chart Image Scraper

**🚀 Now with Browser Pooling Optimization for 70-80% Better Concurrent Performance!**

This MCP server provides tools to fetch TradingView chart images based on ticker and interval with advanced browser pooling for maximum concurrent performance.

## 🔥 Performance Optimizations

### Browser Pooling Technology

- **Production Version**: Uses `main_optimized.py` with browser pooling for maximum concurrent performance
- **Performance Improvement**: 70-80% faster for concurrent requests
- **Concurrent Handling**: Supports up to 4 simultaneous requests with pre-initialized browser instances
- **Expected Performance**:
  - 1 request: ~6-8s (baseline)
  - 2 concurrent: ~3-4s each (60-70% faster)
  - 3 concurrent: ~2.5-3.5s each (70-80% faster)

### Version Comparison

- **`main_optimized.py`** (Production): Browser pooling, concurrent optimization, performance tracking
- **`main.py`** (Legacy): Simple single-browser approach, kept for debugging/fallback

## Features

- **🚀 Optimized Chart Image Tool:** Fetches direct chart images with browser pooling for maximum concurrent performance
- **📊 Performance Statistics:** Built-in performance monitoring and statistics
- **🔄 Browser Pool Management:** Pre-initialized browser instances for zero-overhead concurrent requests
- **🎯 Natural Language Prompts:** Easy chart requests with interval mapping
- **⚙️ Environment Configuration:** Fully configurable via environment variables
- **🔐 TradingView Authentication:** Secure session-based authentication
- **💾 Clipboard Capture:** Direct base64 image capture for faster performance

## Tools

### `get_tradingview_chart_image`

**Description:** Fetches the direct image URL for a TradingView chart snapshot with optimized concurrent performance.

**Performance:** This optimized version achieves 70-80% better performance for concurrent requests using browser pooling technology.

**Arguments:**

- `ticker` (str): The TradingView ticker symbol (e.g., "BYBIT:BTCUSDT.P", "NASDAQ:AAPL"). **Required**.
- `interval` (str): The chart time interval (e.g., '1', '5', '15', '60', '240', 'D', 'W'). **Required**.

**Returns:**

- (str): The direct TradingView snapshot image URL (e.g., `data:image/png;base64,...` or `https://s3.tradingview.com/snapshots/...`).

**Raises:**

- `Error` (MCP type): If the scraper encounters an error. Error codes:
  - `400`: Input error (invalid ticker/interval format)
  - `503`: Scraper error (failure during the scraping process)
  - `500`: Unexpected internal server error

### `get_performance_stats`

**Description:** Get performance statistics for the optimized TradingView MCP server.

**Returns:**

- Detailed metrics about request performance, improvement over baseline, and browser pool status

**Example Output:**

```
🚀 OPTIMIZED TRADINGVIEW MCP SERVER PERFORMANCE STATS
• Total Requests: 12
• Average Time: 3.30s
• Performance Improvement: 70.8%
• Browsers in Pool: 4
• Max Concurrent: 4
```

## Prompts

- `Get the {interval} chart for {ticker}`
  - Maps common timeframe names (e.g., "1 minute", "5 minute", "1 hour", "daily") to TradingView codes
- `Show me the daily TradingView chart for {ticker}`
  - Specifically requests the daily ('D') chart
- `Fetch TradingView chart image for {ticker} on the {interval} timeframe`
  - Comprehensive prompt with timeframe mapping

## Setup

1.  **Create Virtual Environment:**
    ```bash
    # Navigate to the project directory
    cd tradingview-chart-mcp
    # Create the venv (use python3 if python is not linked)
    python3 -m venv .venv
    ```
2.  **Activate Virtual Environment:**

    - **macOS/Linux:**
      ```bash
      source .venv/bin/activate
      ```
    - **Windows (Git Bash/WSL):**
      ```bash
      source .venv/Scripts/activate
      ```
    - **Windows (Command Prompt):**
      ```bash
      .venv\\Scripts\\activate.bat
      ```
    - **Windows (PowerShell):**
      ```bash
      .venv\\Scripts\\Activate.ps1
      ```
      _(Note: You might need to adjust PowerShell execution policy: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`)_

    Your terminal prompt should now indicate you are in the `(.venv)`.

3.  **Install Dependencies (inside venv):**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Configure Environment (for Local Testing):**
    - Copy `.env.example` to `.env`.
    - Fill in your `TRADINGVIEW_SESSION_ID` and `TRADINGVIEW_SESSION_ID_SIGN` in the `.env` file. You can obtain these from your browser's cookies after logging into TradingView.
    - This `.env` file is used when running the server directly (e.g., `python main.py`) for local testing.
    - Adjust optional scraper settings (`MCP_SCRAPER_HEADLESS`, etc.) in `.env` if needed for local runs.
5.  **Ensure ChromeDriver:** Make sure `chromedriver` is installed and accessible in your system's PATH, or configure the `tview-scraper.py` accordingly if it allows specifying a path.

## Running the Server

### Production (Optimized)

```bash
python main_optimized.py
```

### Legacy (Simple)

```bash
python main.py
```

### Command Line Options

```bash
# HTTP transport with custom port
python main_optimized.py --transport streamable-http --port 8003

# Enable authentication (default for TradingView)
python main_optimized.py --auth

# Adjust concurrency
python main_optimized.py --max-concurrent 6

# Disable browser pooling (fallback to traditional)
python main_optimized.py --disable-pooling
```

## 🧪 Performance Testing

### Agent-Style Testing

Test the server using the same approach as the production agent:

```bash
cd tests/

# Sequential performance test
python test_mcp_agent_style.py --runs 5 --ticker BYBIT:BTCUSDT.P --interval 240

# Concurrent performance test
python test_mcp_agent_style.py --concurrent 3 --ticker BYBIT:BTCUSDT.P --interval 240

# Test different symbols and timeframes
python test_mcp_agent_style.py --concurrent 4 --ticker NASDAQ:AAPL --interval 15
python test_mcp_agent_style.py --concurrent 2 --ticker BYBIT:ETHUSDT.P --interval D
```

### Expected Results

- **Sequential**: ~6-8s per request (baseline)
- **Concurrent (3x)**: ~2.5-3.5s per request (70-80% improvement)
- **Success Rate**: 100% reliability under load
- **Throughput**: Up to 4 concurrent requests efficiently handled

## 🔧 Technical Details

### Browser Pooling Architecture

- **Pre-initialized Browsers**: 4 browser instances ready for immediate use
- **Thread-Safe Pool**: Concurrent access with proper locking
- **Async Semaphore**: Optimal request limiting
- **Performance Tracking**: Real-time statistics and monitoring
- **Graceful Cleanup**: Proper browser lifecycle management

### Save Shortcut Feature

The `MCP_SCRAPER_USE_SAVE_SHORTCUT` feature allows you to capture chart images directly to the clipboard as base64 data URLs:

**Benefits:**

- **Faster Performance**: No HTTP requests needed
- **More Reliable**: No dependency on TradingView's CDN
- **Offline Capability**: Works once chart is loaded
- **Direct Integration**: Base64 data URLs for immediate use

**Configuration:**

```bash
# Enable clipboard image capture (DEFAULT)
MCP_SCRAPER_USE_SAVE_SHORTCUT=True

# Disable and use traditional screenshot links
MCP_SCRAPER_USE_SAVE_SHORTCUT=False
```

## Usage Examples

**Available Tools:**

- `get_tradingview_chart_image(ticker: str, interval: str)`: Optimized chart fetching with browser pooling

**Example Prompts:**

- "Get the 15 minute chart for NASDAQ:AAPL"
- "Show me the daily chart for BYBIT:BTCUSDT.P"
- "Fetch TradingView chart image for COINBASE:ETHUSD on the 60 timeframe"

## Deactivating the Virtual Environment

When finished:

```bash
deactivate
```

## 🔄 Fallback Options

If you encounter issues with the optimized version:

1. **Disable Browser Pooling:**

   ```bash
   python main_optimized.py --disable-pooling
   ```

2. **Use Legacy Version:**

   ```bash
   python main.py
   ```

3. **Debug Mode:**
   ```bash
   python main_optimized.py --log-level DEBUG
   ```

## 📊 Performance Monitoring

The optimized server includes built-in performance monitoring:

- Request success rates
- Average response times
- Performance improvement metrics
- Browser pool utilization
- Concurrent request handling statistics

Access these metrics via the `get_performance_stats` tool or through the server logs.

## 🔌 Using with MCP Clients (Claude Desktop / Cursor)

This server supports two ways of providing configuration:

1.  **Via `.env` file (for local testing):** When running `python main.py` directly, the server will load credentials and settings from a `.env` file in the project directory.
2.  **Via Client Environment Variables (Recommended for Integration):** When run by an MCP client (like Claude/Cursor), you should configure the client to inject the required environment variables directly. **These will override any values found in a `.env` file.**

### Claude Desktop

1.  Open your Claude Desktop configuration file:
    - **Windows:** `%APPDATA%\\Claude\\claude_desktop_config.json`
    - **macOS:** `~/Library/Application\ Support/Claude/claude_desktop_config.json`
2.  Add or merge the following within the `mcpServers` object. Provide your credentials in the `env` block:

    ```json
    {
      "mcpServers": {
        "tradingview-chart-mcp": {
          "command": "/absolute/path/to/your/tradingview-chart-mcp/.venv/bin/python3",
          "args": ["/absolute/path/to/your/tradingview-chart-mcp/main.py"],
          "env": {
            "TRADINGVIEW_SESSION_ID": "YOUR_SESSION_ID_HERE",
            "TRADINGVIEW_SESSION_ID_SIGN": "YOUR_SESSION_ID_SIGN_HERE"
            // Optional: Add MCP_SCRAPER_* variables here too if needed
            // "MCP_SCRAPER_HEADLESS": "False"
          }
        }
        // ... other servers if any ...
      }
    }
    ```

3.  Replace the placeholder paths (`command`, `args`) with your actual absolute paths.
4.  Replace `YOUR_SESSION_ID_HERE` and `YOUR_SESSION_ID_SIGN_HERE` with your actual TradingView credentials.
5.  Restart Claude Desktop.

### Cursor

1.  Go to: `Settings -> Cursor Settings -> MCP -> Edit User MCP Config (~/.cursor/mcp.json)`.
2.  Add or merge the following within the `mcpServers` object. Provide your credentials in the `env` block:

    ```json
    {
      "mcpServers": {
        "tradingview-chart-mcp": {
          "command": "/absolute/path/to/your/tradingview-chart-mcp/.venv/bin/python3",
          "args": ["/absolute/path/to/your/tradingview-chart-mcp/main.py"],
          "env": {
            "TRADINGVIEW_SESSION_ID": "YOUR_SESSION_ID_HERE",
            "TRADINGVIEW_SESSION_ID_SIGN": "YOUR_SESSION_ID_SIGN_HERE"
            // Optional: Add MCP_SCRAPER_* variables here too if needed
            // "MCP_SCRAPER_HEADLESS": "False"
          }
        }
        // ... other servers if any ...
      }
    }
    ```

3.  Replace the placeholder paths (`command`, `args`) with your actual absolute paths.
4.  Replace `YOUR_SESSION_ID_HERE` and `YOUR_SESSION_ID_SIGN_HERE` with your actual TradingView credentials.
5.  Restart Cursor.

### Installing via Smithery

To install TradingView Chart Image Scraper for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@ertugrul59/tradingview-chart-mcp):

```bash
npx -y @smithery/cli install @ertugrul59/tradingview-chart-mcp --client claude
```

## Configuration

### Environment Variables

The following environment variables can be set to configure the scraper:

- `TRADINGVIEW_SESSION_ID`: Your TradingView session ID (required)
- `TRADINGVIEW_SESSION_ID_SIGN`: Your TradingView session ID signature (required)
- `MCP_SCRAPER_HEADLESS`: Run browser in headless mode (default: `True`)
- `MCP_SCRAPER_WINDOW_WIDTH`: Browser window width (default: `1400`)
- `MCP_SCRAPER_WINDOW_HEIGHT`: Browser window height (default: `1400`)
- `MCP_SCRAPER_USE_SAVE_SHORTCUT`: Use clipboard image capture instead of screenshot links (default: `True`)
- `MCP_SCRAPER_CHART_PAGE_ID`: Your TradingView chart layout ID (**required** - see [Chart Layout Setup](#chart-layout-setup))

### Save Shortcut Feature

The `MCP_SCRAPER_USE_SAVE_SHORTCUT` feature allows you to capture chart images directly to the clipboard as base64 data URLs instead of getting screenshot links. This eliminates the need to download images from URLs.

**Benefits:**

- Faster chart capture (no HTTP requests needed)
- More reliable (no dependency on TradingView's CDN)
- Works offline once the chart is loaded
- Direct base64 data URLs for immediate use

**How it works:**

- When enabled (`True`): Uses `Shift+Ctrl+S` (or `Shift+Cmd+S` on Mac) to capture chart image directly to clipboard
- When disabled (`False`): Uses traditional `Alt+S` to get screenshot links, then converts to image URLs

**Configuration:**

```bash
# Enable clipboard image capture (DEFAULT)
MCP_SCRAPER_USE_SAVE_SHORTCUT=True

# Disable and use traditional screenshot links
MCP_SCRAPER_USE_SAVE_SHORTCUT=False
```

### Chart Layout Setup

The `MCP_SCRAPER_CHART_PAGE_ID` is **required** and must be set to your own TradingView chart layout ID. This ensures you have full control over the chart appearance and indicators.

#### Step 1: Create a Chart Layout on TradingView

1. Go to [TradingView](https://www.tradingview.com/chart/)
2. Configure your chart with your preferred settings:
   - **Chart type**: Candlesticks (recommended for technical analysis)
   - **Indicators**: Add any indicators you want (e.g., VWAP, Volume)
   - **Theme**: Dark or light background
   - **Volume pane**: Resize to your preference (25-30% recommended)

#### Step 2: Save Your Layout

1. Click the **Layout dropdown** (top-left corner)
2. Click **Save layout as...** or **Rename**
3. Give it a descriptive name (e.g., "Trading Analysis")

#### Step 3: Get Your Chart Page ID

Look at the URL in your browser after saving:

```
https://www.tradingview.com/chart/AbCdEfGh/?symbol=...
                                 ^^^^^^^^
                                 This is your Chart Page ID
```

Copy the 8-character ID (e.g., `AbCdEfGh`).

#### Step 4: Configure the Environment Variable

Add to your `.env` file or MCP client config:

```bash
MCP_SCRAPER_CHART_PAGE_ID=AbCdEfGh
```

#### Recommended Layout for AI Analysis

For optimal AI chart analysis (e.g., Wyckoff method), consider:

| Component | Recommendation | Why |
|-----------|----------------|-----|
| **Chart type** | Standard Candlesticks | True wicks visible for pattern recognition |
| **Volume** | 25-30% of chart height | Essential for volume analysis |
| **VWAP** | Single line (optional) | Institutional reference point |
| **Background** | High contrast (dark/light) | Better AI readability |
| **Grid** | Disabled or minimal | Reduces visual noise |

**Avoid:**
- Too many overlapping indicators
- Heikin-Ashi candles (hide true price action)
- Small volume pane
- Busy color schemes
