import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"data.csv")

df['Swing_High'] = df['close'][
    (df['close'] > df['close'].shift(1)) &
    (df['close'] > df['close'].shift(-1))
]

df['Swing_Low'] = df['close'][
    (df['close'] < df['close'].shift(1)) &
    (df['close'] < df['close'].shift(-1))
]

plt.figure(figsize=(12, 6))
plt.plot(df['close'], label='Close Price', linewidth=2)


plt.scatter(df.index, df['Swing_High'], marker='.', label='Swing High', s=30, color = "red")
plt.scatter(df.index, df['Swing_Low'], marker='.', label='Swing Low', s=30, color = "green")


plt.title("Close Price with Swing Highs & Lows")
plt.xlabel("Time/Minute")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("close_price_with_swings.png")
plt.show()
