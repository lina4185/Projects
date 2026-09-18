import pandas as pd
import numpy as np

rows = 1000

np.random.seed(42)


data = {
    "temperature": np.random.randint(-5, 35, rows),
    "wind_speed": np.random.uniform(0, 15, rows),
    "traffic_level": np.random.randint(0, 3, rows),
    "battery_health": np.random.uniform(0.7, 1.0, rows),
    "driver_aggressiveness": np.random.uniform(0, 1, rows),
    "actual_range": np.random.uniform(180, 400, rows)
}


df = pd.DataFrame(data)


df.to_csv("fake_dataset.csv", index=False)

print(df.head())