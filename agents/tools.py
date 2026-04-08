from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize shared clients for our tools
tavily_client = TavilySearchResults(max_results=5)

# Mock databases simulating internal market data systems
mock_fundamentals_db = {
    "aapl": "P/E: 28.5, Revenue Growth: 12% YoY, Market Cap: $3.2T",
    "nvda": "P/E: 65.2, Revenue Growth: 122% YoY, Market Cap: $3.1T",
    "tsla": "P/E: 72.1, Revenue Growth: -8% YoY, Market Cap: $780B",
}
mock_ta_db = {
    "aapl": "RSI: 62 (neutral), MACD: bullish crossover, 50-day MA: above 200-day MA",
    "nvda": "RSI: 71 (overbought), MACD: bearish divergence, Volume: declining",
}


@tool
def market_news_search(query: str) -> str:
    """Searches financial news for market-moving events and analyst opinions."""
    return tavily_client.invoke(f"site:reuters.com OR site:bloomberg.com {query}")


@tool
def sec_filing_search(query: str) -> str:
    """Searches SEC filings and earnings reports for fundamental data."""
    return tavily_client.invoke(f"site:sec.gov {query} 10-K OR 10-Q")


@tool
def stock_fundamentals_lookup(ticker: str) -> str:
    """Looks up fundamental data for a stock from our internal database."""
    return mock_fundamentals_db.get(ticker.lower(), f"Ticker '{ticker}' not found.")


@tool
def technical_indicators_lookup(ticker: str) -> str:
    """Retrieves technical analysis indicators for a given stock."""
    return mock_ta_db.get(ticker.lower(), f"No TA data for '{ticker}'.")


all_tools = [
    market_news_search,
    sec_filing_search,
    stock_fundamentals_lookup,
    technical_indicators_lookup,
]
print("Financial Toolkit defined successfully.")
stock_fundamentals_lookup.invoke("AAPL")
