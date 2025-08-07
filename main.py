import sys
import subprocess
import argparse
from abc import ABC, abstractmethod
import os
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline
import finnhub

class ScriptRunner(ABC):
    """Abstract base for external script runners."""
    def __init__(self, script_path: str):
        self.script_path = script_path

    @abstractmethod
    def run(self):
        pass

class TrendLinesRunner(ScriptRunner):
    def __init__(self):
        super().__init__('trendlines.py')

    def run(self):
        subprocess.run([sys.executable, self.script_path], check=True)

class SentimentRunner(ScriptRunner):
    def __init__(self):
        super().__init__('sentiment.py')

    def run(self):
        subprocess.run([sys.executable, self.script_path], check=True)

class NeuroTraderRunner(ScriptRunner):
    def __init__(self):
        super().__init__('neurotrader888.py')

    def run(self):
        subprocess.run([sys.executable, self.script_path], check=True)

class ArimaRunner(ScriptRunner):
    def __init__(self):
        super().__init__('ARIMA.py')

    def run(self):
        subprocess.run([sys.executable, self.script_path], check=True)

class DataCollectorRunner:
    """
    Collects index constituents and selects stocks based on liquidity and data availability.
    """
    def __init__(self, index: str, min_volume: float, start_date: str, end_date: str):
        self.index = index
        self.min_volume = min_volume
        self.start_date = start_date
        self.end_date = end_date

    def run(self):
        if self.index.upper() == 'SP500':
            df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        else:
            tables = pd.read_html('https://en.wikipedia.org/wiki/NASDAQ-100')
            table = next(t for t in tables if 'Ticker' in t.columns)
            tickers = table['Ticker'].str.replace('.', '-', regex=False).tolist()

        print(f"Fetched {len(tickers)} tickers for {self.index}")

        data = yf.download(tickers, start=self.start_date, end=self.end_date)
        adj = data['Adj Close']
        vol = data['Volume']

        avg_vol = vol.mean()
        liq_tickers = avg_vol[avg_vol >= self.min_volume].index.tolist()

        avail = adj.notna().all()
        avail_tickers = avail[avail].index.tolist()

        selected = sorted(set(liq_tickers) & set(avail_tickers))
        print(f"Selected {len(selected)} tickers after filters.")
        return selected

class AnalysisRunner:
    """
    Runs data collection then computes signals:
      - Momentum
      - Volatility
      - Valuation (P/E)
      - Standardization
      - ARIMA predictions
      - Sentiment via FinBERT with news fetched automatically
    """
    def __init__(self, index, min_volume, start, end, window, finnhub_key):
        self.index = index
        self.min_volume = min_volume
        self.start = start
        self.end = end
        self.window = window
        self.finnhub_key = finnhub_key or os.getenv("FINNHUB_API_KEY")
        if not self.finnhub_key:
            raise ValueError("Finnhub API key must be provided via --finnhub-key or FINNHUB_API_KEY env var")
        self.fh_client = finnhub.Client(api_key=self.finnhub_key)

    def run(self):
        # 1) Collect tickers
        collector = DataCollectorRunner(self.index, self.min_volume, self.start, self.end)
        tickers = collector.run()

        # 2) Fetch price data
        data = yf.download(tickers, start=self.start, end=self.end)
        price = data['Adj Close']
        vol = data['Volume']

        # 3) Momentum
        momentum = price.pct_change(self.window)
        print("\\nMomentum (last rows):")
        print(momentum.tail())

        # 4) Volatility
        returns = price.pct_change()
        volatility = returns.rolling(self.window).std()
        print("\\nVolatility (last rows):")
        print(volatility.tail())

        # 5) Valuation via yfinance info trailingPE
        pe = {}
        for tkr in tickers:
            info = yf.Ticker(tkr).info
            pe[tkr] = info.get('trailingPE', np.nan)
        valuation = pd.Series(pe)
        print("\\nValuation (P/E):")
        print(valuation)

        # 6) Standardization
        mom_std = (momentum - momentum.mean()) / momentum.std()
        vol_std = (volatility - volatility.mean()) / volatility.std()
        print("\\nStandardized Momentum (last rows):")
        print(mom_std.tail())
        print("\\nStandardized Volatility (last rows):")
        print(vol_std.tail())

        # 7) ARIMA predictions
        arima_preds = {}
        for tkr in tickers:
            series = price[tkr].dropna()
            try:
                model = ARIMA(series, order=(1,0,1)).fit()
                pred = model.forecast(steps=1)
                arima_preds[tkr] = pred.iloc[0]
            except Exception:
                arima_preds[tkr] = np.nan
        arima_series = pd.Series(arima_preds)
        print("\\nARIMA 1-step forecast:")
        print(arima_series)

        # 8) Auto-fetch news and sentiment
        sentiment_scores = {}
        for tkr in tickers:
            try:
                news = self.fh_client.company_news(tkr, _from=self.start, to=self.end)
                headlines = [item['headline'] for item in news]
                if headlines:
                    nlp = pipeline('sentiment-analysis', model='ProsusAI/finbert')
                    scores = [nlp(txt)[0]['score'] for txt in headlines]
                    sentiment_scores[tkr] = np.mean(scores)
                else:
                    sentiment_scores[tkr] = np.nan
            except Exception:
                sentiment_scores[tkr] = np.nan
        sentiment_series = pd.Series(sentiment_scores)
        print("\\nAuto-fetched Sentiment (avg FinBERT score):")
        print(sentiment_series)



parser = argparse.ArgumentParser(
    description="Run analysis scripts, collect stocks, or compute signals"
)
parser.add_argument(
    'script',
    choices=['collect','analyze'],
    help="Operation to perform"
)
parser.add_argument('--index', choices=['SP500','NASDAQ100'], default='SP500')
parser.add_argument('--min-volume', type=float, default=1e7)
parser.add_argument('--start', default='2020-01-01')
parser.add_argument('--end', default=None)
parser.add_argument('--window', type=int, default=20,
                    help="Window size for momentum/volatility")
parser.add_argument('--news', nargs='+',
                    help="News headlines for FinBERT sentiment")

args = parser.parse_args()
end_date = args.end or pd.Timestamp.today().strftime('%Y-%m-%d')

if args.script == 'collect':
    runner = DataCollectorRunner(args.index, args.min_volume, args.start, end_date)
elif args.script == 'analyze':
    runner = AnalysisRunner(args.index, args.min_volume, args.start,
                            end_date, args.window, args.news)
else:
    parser.print_help()
    sys.exit(1)

runner.run()
