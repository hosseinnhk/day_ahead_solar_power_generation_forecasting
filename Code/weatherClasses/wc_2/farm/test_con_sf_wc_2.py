import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_1 = pd.read_csv('lightgbm_pred_0_148.csv', parse_dates=['date'], index_col=['date', 'time'])
df_2 = pd.read_csv('lightgbm_pred_148_296.csv', parse_dates=['date'], index_col=['date', 'time'])
df_3 = pd.read_csv('lightgbm_pred_296_444.csv', parse_dates=['date'], index_col=['date', 'time'])
df_4 = pd.read_csv('lightgbm_pred_444_592.csv', parse_dates=['date'], index_col=['date', 'time'])
df_5 = pd.read_csv('lightgbm_pred_592_740.csv', parse_dates=['date'], index_col=['date', 'time'])

df = pd.DataFrame(pd.concat((df_1, df_2, df_3, df_4, df_5), axis=0))
# df = df.reset_index(drop=False)
# print(df.columns)
df = df.drop(['Unnamed: 0'], axis=1)
df.to_csv('reg_preds_wc_2.csv')
