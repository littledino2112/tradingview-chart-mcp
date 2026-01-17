# TradingView Chart MCP - Project Assessment

**Date:** January 17, 2026
**Assessed by:** Claude Opus 4.5

---

## Overview

This document provides a comprehensive assessment of the TradingView Chart MCP project, identifying strengths and areas for significant improvement.

---

## What's Done Well

### 1. Browser Pooling Architecture

The `OptimizedTradingViewMCPServer` is well-designed. Pre-initializing browser instances eliminates the 3-4s creation overhead per request, achieving **70-80% performance improvement** for concurrent requests.

**Key implementation details:**
- Thread-safe pool management with `threading.Lock`
- Async request throttling with `asyncio.Semaphore`
- Configurable pool size (default: 4 instances)

```python
# Pool management is thread-safe
def _get_browser(self) -> Optional[TradingViewScraper]:
    with self.browser_lock:
        if self.browser_pool:
            return self.browser_pool.pop()
        return None
```

### 2. Dual Capture Methods with Smart Defaults

Having both capture methods provides reliability:

| Method | Mechanism | Speed | Reliability |
|--------|-----------|-------|-------------|
| **Save Shortcut** (default) | `Shift+Ctrl/Cmd+S` → clipboard image | Faster | No CDN dependency |
| **Traditional** | `Alt+S` → share link → S3 URL | Slower | More compatible |

The save shortcut as default is the right choice - it's faster, doesn't depend on TradingView's CDN, and works offline after page load.

### 3. Intelligent Waiting Strategy

The waiting system uses parallel element detection with method-specific optimizations:

- **Parallel detection:** Uses Selenium's `any_of()` for multiple selectors
- **Method-specific waits:**
  - Save shortcut: Ultra-minimal wait (1.5s loading check)
  - Traditional: Data validation wait (2s max)
- **Aggressive timeouts:** 6s chart wait, 3s clipboard wait

### 4. Error Handling with Retry Logic

The exception hierarchy enables targeted retry behavior:

```
TradingViewScraperError (base)
├── TradingViewClipboardServerError (retryable)
│   - Triggered by: 40001, 50000, 502, 503
│   - Auto-retry up to 5 times
└── TradingViewServerError (general)
```

Server errors detected in clipboard JSON responses trigger automatic retries with appropriate backoff.

### 5. Clean MCP Integration

- Returns `Image` objects via FastMCP for proper multimodal support
- Natural language prompts with interval mapping
- Performance stats endpoint for monitoring

---

## Significant Improvements Needed

### 1. No Graceful Recovery from Browser Pool Exhaustion

**Problem:** If a browser crashes or becomes unresponsive, the pool shrinks permanently. There's no mechanism to detect unhealthy browsers or replenish the pool.

**Current behavior:**
```python
# Browser is always returned to pool, even if it failed
finally:
    if scraper:
        self._return_browser(scraper)
```

**Impact:** Over time, the pool could fill with dead/stuck browsers, degrading performance or causing failures.

**Recommendation:** Add health checks before returning to pool, and background replenishment:
```python
def _return_browser(self, scraper, healthy=True):
    if healthy:
        with self.browser_lock:
            self.browser_pool.append(scraper)
    else:
        scraper.close()
        self._spawn_replacement_browser()  # Async replenishment
```

**Priority:** High
**Effort:** Medium

---

### 2. Session Cookie Expiration Handling

**Problem:** TradingView session cookies expire, but there's no detection or refresh mechanism. Once expired, all requests silently fail or return incomplete charts (no premium indicators).

**Impact:** Users won't know why their charts stopped working until they manually refresh cookies.

**Recommendation:**
- Detect authentication failures (check for "Sign In" button or subscription warnings)
- Add health-check endpoint or startup validation
- Consider automatic re-authentication if credentials are stored

**Priority:** High
**Effort:** Medium

---

### 3. No Request Queuing or Backpressure

**Problem:** When all 4 browsers are busy, new requests block on the semaphore indefinitely. There's no timeout, queue depth limit, or feedback to the caller.

**Current behavior:**
```python
async with self.semaphore:  # Blocks forever if pool exhausted
    # ... process request
```

**Impact:** Under high load, requests pile up with no visibility, leading to timeouts at the MCP client level without useful error messages.

**Recommendation:** Add request timeout and queue monitoring:
```python
try:
    async with asyncio.timeout(30):  # Request-level timeout
        async with self.semaphore:
            # ... process
except asyncio.TimeoutError:
    raise TradingViewScraperError("Request timed out - server at capacity")
```

**Priority:** Medium
**Effort:** Low

---

### 4. Memory Leak Risk with Long-Running Browsers

**Problem:** Pooled browsers are never recycled. Chrome accumulates memory over time, especially with complex TradingView charts.

**Impact:** In long-running deployments, memory usage grows unbounded, eventually causing OOM kills or severe slowdowns.

**Recommendation:** Implement browser recycling after N requests or based on memory threshold:
```python
class PooledBrowser:
    def __init__(self, scraper):
        self.scraper = scraper
        self.request_count = 0
        self.max_requests = 100  # Recycle after 100 requests

    def needs_recycle(self):
        return self.request_count >= self.max_requests
```

**Priority:** Medium
**Effort:** Medium

---

### 5. Test Coverage is Minimal and Manual

**Problem:** The tests in `tests/` are performance benchmarks requiring manual execution and TradingView credentials. There are no unit tests, no mocking, and no CI integration.

**Current tests:**
- `test_mcp_agent_style.py` - Manual performance testing
- `test_timeframes.py` - Manual timeframe validation

**Impact:** Regressions can easily slip in. Code changes have no automated verification.

**Recommendation:**
- Add unit tests with mocked WebDriver for scraper logic
- Add integration tests with a test TradingView account
- Add CI pipeline (GitHub Actions) for syntax/import validation at minimum

**Priority:** Medium
**Effort:** High

---

### 6. Hardcoded Chart Page ID

**Problem:** The default chart page ID (`XHDbt5Yy`) is hardcoded and represents someone's personal saved chart layout.

```python
DEFAULT_CHART_PAGE_ID = "XHDbt5Yy"
```

**Impact:** Single point of failure. If that chart is deleted or modified, the service breaks for everyone using defaults.

**Recommendation:**
- Use TradingView's generic chart URL format without a saved layout ID
- Or document this clearly and make it a required configuration

**Priority:** Low
**Effort:** Low

---

### 7. Synchronous Browser Operations Block the Event Loop

**Problem:** While MCP tools are async, actual scraping uses `run_in_executor()` with the default ThreadPoolExecutor:

```python
result = await asyncio.get_event_loop().run_in_executor(
    None, scraper.get_chart_image_url, ticker, interval
)
```

**Impact:** The default executor has limited workers, potentially bottlenecking concurrent requests beyond its thread count.

**Recommendation:** Use a dedicated ThreadPoolExecutor sized to match the browser pool:
```python
self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)
# Then use: await loop.run_in_executor(self.executor, ...)
```

**Priority:** Low
**Effort:** Low

---

## Summary

| Category | Rating | Notes |
|----------|--------|-------|
| **Architecture** | Strong | Browser pooling, dual methods, clean MCP integration |
| **Performance** | Good | Aggressive optimizations, intelligent waiting |
| **Reliability** | Needs Work | No pool recovery, no session validation, no backpressure |
| **Maintainability** | Needs Work | No unit tests, hardcoded defaults, memory leak risk |
| **Production Readiness** | 70% | Works well short-term, needs resilience for long-running deployment |

---

## Improvement Priority Matrix

| Improvement | Priority | Effort | Impact |
|-------------|----------|--------|--------|
| Browser pool recovery | High | Medium | Prevents silent degradation |
| Session cookie validation | High | Medium | Prevents auth failures |
| Request timeout/backpressure | Medium | Low | Better error handling |
| Browser memory recycling | Medium | Medium | Prevents OOM in long-running |
| Automated tests | Medium | High | Prevents regressions |
| Remove hardcoded chart ID | Low | Low | Removes single point of failure |
| Dedicated thread executor | Low | Low | Minor concurrency improvement |

---

## Next Steps

1. Review this assessment and prioritize improvements
2. Create feature branches for selected improvements
3. Implement changes incrementally with testing
4. Update documentation as features are added
