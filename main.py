import sys
import subprocess
import argparse
from abc import ABC, abstractmethod

class ScriptRunner(ABC):
    
    def __init__(self, script_path: str):
        # Path to the external script
        self.script_path = script_path

    @abstractmethod
    def run(self):
        #Executes the external script.

        pass

class TrendLinesRunner(ScriptRunner):
    def __init__(self):
        super().__init__('trendlines.py')  # access trendlines.py directly

    def run(self):
        # Calls the trendlines script
        subprocess.run([sys.executable, self.script_path], check=True)

class SentimentRunner(ScriptRunner):
    def __init__(self):
        super().__init__('sentiment.py')  # access sentiment.py directly

    def run(self):
        # Calls the sentiment analysis script
        subprocess.run([sys.executable, self.script_path], check=True)

class NeuroTraderRunner(ScriptRunner):
    def __init__(self):
        super().__init__('neurotrader888.py')  # access neurotrader888.py directly

    def run(self):
        # Calls the neurotrader888 script
        subprocess.run([sys.executable, self.script_path], check=True)

class ArimaRunner(ScriptRunner):
    def __init__(self):
        super().__init__('ARIMA.py')  # access ARIMA.py directly

    def run(self):
        # Calls the ARIMA forecasting script
        subprocess.run([sys.executable, self.script_path], check=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    'script'
)
args = parser.parse_args()

if args.script == 'trendlines':
    runner = TrendLinesRunner()
elif args.script == 'sentiment':
    runner = SentimentRunner()
elif args.script == 'neuro':
    runner = NeuroTraderRunner()
elif args.script == 'arima':
    runner = ArimaRunner()
else:
    parser.print_help()
    sys.exit(1)

# Execute the selected external program
runner.run()
