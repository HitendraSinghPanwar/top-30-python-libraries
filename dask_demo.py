#Dask

import dask.dataframe as dd

df = dd.read_csv(r"C:\Users\thaku\Downloads\dataset.csv")

numeric_df = df.select_dtypes(include='number')

mean_value = numeric_df.mean().compute()

print("average value:\n", mean_value )