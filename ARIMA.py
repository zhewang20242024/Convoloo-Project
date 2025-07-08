import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


p = int(input("Enter AR order (p): "))
d = int(input("Enter differencing order (d): "))
q = int(input("Enter MA order (q): "))
N = int(input("Enter forecast steps (N): "))


df = pd.read_csv("C:\Project\Modeling\data.csv")
y = df['close'].astype(float)

model = ARIMA(y, order=(p, d, q))
model_fit = model.fit()


print(model_fit.summary())

forecast = model_fit.get_forecast(steps=N)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(range(len(y), len(y) + N), forecast_mean, label="Forecast", color="orange")
plt.fill_between(range(len(y), len(y) + N),
                    conf_int.iloc[:, 0],
                    conf_int.iloc[:, 1],
                    color='gray', alpha=0.3, label="95% CI")

plt.title(f"ARIMA({p},{d},{q}) Forecast ({N} Steps Ahead)")
plt.xlabel("Time/Minute")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
