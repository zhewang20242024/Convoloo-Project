import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2


lookback    = int(input("Enter look-back periods: "))
lookforward = int(input("Enter look-forward periods: "))
k           = int(input("Enter number of clusters for S/R levels: "))


window = lookback + lookforward + 1

df = pd.read_csv("C:\Project\Modeling\data.csv")

price = df['close']


roll_max = price.rolling(window, center=True).max()
roll_min = price.rolling(window, center=True).min()
highs    = price.where(price == roll_max).dropna()
lows     = price.where(price == roll_min).dropna()


def cluster_levels(levels, k):
    vals      = levels.values.reshape(-1,1)
    centroids, _ = kmeans2(vals, k, minit='points')
    return sorted(centroids.flatten())

res_levels = cluster_levels(highs, k)
sup_levels = cluster_levels(lows,  k)


plt.figure(figsize=(12,6))
plt.plot(price, label='Close')
plt.scatter(highs.index, highs, marker='.', label='Swing Highs', color = "red")
plt.scatter(lows.index,  lows,  marker='.',    label='Swing Lows', color = "green")
for r in res_levels:
    plt.hlines(r, xmin=price.index.min(), xmax=price.index.max(),
                linestyles='--', label=f'Resistance ~{r:.2f}', color = "cyan")
for s in sup_levels:
    plt.hlines(s, xmin=price.index.min(), xmax=price.index.max(),
                linestyles='-.', label=f'Support ~{s:.2f}', color = "magenta")
plt.title(f"S/R Levels (lookback={lookback}, lookforward={lookforward}, k={k})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
