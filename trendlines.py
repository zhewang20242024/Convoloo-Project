import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lookback    = int(input("Enter look-back periods: "))
lookforward = int(input("Enter look-forward periods: "))
window_size = int(input("Enter analysis window size: "))


swing_window = lookback + lookforward + 1


df = pd.read_csv("C:\Project\Modeling\data.csv")
price = df['close']


plt.figure(figsize=(12, 6))
plt.plot(price.index, price.values, label='Close Price', linewidth=1)


cmap = plt.get_cmap('tab10')
n_segments = (len(price) + window_size - 1) // window_size

for i in range(n_segments):
    start = i * window_size
    end   = min(start + window_size, len(price))
    segment = price.iloc[start:end]

    if len(segment) < swing_window:
        continue

    roll_max = segment.rolling(swing_window, center=True).max()
    roll_min = segment.rolling(swing_window, center=True).min()
    highs = segment.where(segment == roll_max).dropna()
    lows  = segment.where(segment == roll_min).dropna()

    if len(highs) >= 2:
        xh = highs.index.astype(float)
        yh = highs.values
        mh, bh = np.polyfit(xh, yh, 1)
        idx = np.arange(start, end, dtype=float)
        upper = mh * idx + bh
        if i ==0:
            plt.plot(idx, upper,
                    linestyle='--',
                    color="red",
                    label='Upper Trend seg')
        else :
                plt.plot(idx, upper,
                    linestyle='--',
                    color="red")

    if len(lows) >= 2:
        xl = lows.index.astype(float)
        yl = lows.values
        ml, bl = np.polyfit(xl, yl, 1)
        idx = np.arange(start, end, dtype=float)
        lower = ml * idx + bl
        if i == 0:
            plt.plot(idx, lower,
                    linestyle='--',
                    color="green",
                    label='Lower Trend seg')
        else :
                plt.plot(idx, upper,
                    linestyle='--',
                    color="red")
    if i == 0:
        plt.scatter(xh, yh, marker='.', c='red',   label='High Extremes')
        plt.scatter(xl, yl, marker='.', c='green', label='Low Extremes')
    else :
        plt.scatter(xh, yh, marker='.', c='red')
        plt.scatter(xl, yl, marker='.', c='green')
plt.title(f"Windowed Trend Lines (window_size={window_size}, lookback={lookback}, lookforward={lookforward})")
plt.xlabel("Data Index")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
