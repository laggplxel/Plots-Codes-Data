import pandas as pd

CSV_PATH = "data/Lchi3_slope_gaps_1_10000.csv"

df = pd.read_csv(CSV_PATH)

total_flips = df['flip'].sum()
total_gaps = len(df)
flip_prob = df['flip'].mean()

print(f"Total gaps: {total_gaps}")
print(f"Total flips: {total_flips}")
print(f"Overall flip probability: {total_flips/total_gaps:.6%}")
print(f"Average flip value: {flip_prob:.6f}")