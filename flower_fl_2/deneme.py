import pandas as pd
import os

for client_id in range(1, 11):
    df = pd.read_csv(os.path.join("client_datasets", f"client_data_{client_id}.csv"))
    print(f"Client {client_id}:")
    print(f"  Min flow: {df['flow'].min()}")
    print(f"  Max flow: {df['flow'].max()}")
    print(f"  Mean flow: {df['flow'].mean()}")
    print("-" * 30)
