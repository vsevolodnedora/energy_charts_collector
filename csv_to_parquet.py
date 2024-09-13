"""
CSV files exceed the maximum github allowed size,
so we provide instead parquet files that can be
read with pandas as csv files
"""

import pandas as pd
from glob import glob

if __name__ == '__main__':
    files = glob("./data/*.csv")
    for file in files:
        print(f"Processing {file}")
        df = pd.read_csv(file,index_col=0)
        df.to_parquet(file.replace('.csv', '.parquet'))