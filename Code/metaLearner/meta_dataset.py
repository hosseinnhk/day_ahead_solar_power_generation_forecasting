import numpy as np
import pandas as pd

df_1 = pd.read_csv('reg_preds_wc_0.csv', parse_dates=['date'], index_col=['date', 'time'])
df_2 = pd.read_csv('reg_preds_wc_1.csv', parse_dates=['date'], index_col=['date', 'time'])
df_3 = pd.read_csv('reg_preds_wc_2.csv', parse_dates=['date'], index_col=['date', 'time'])
df_4 = pd.read_csv('reg_preds_wc_3.csv', parse_dates=['date'], index_col=['date', 'time'])
df_5 = pd.read_csv('reg_preds_wc_oth.csv', parse_dates=['date'], index_col=['date', 'time'])

df = pd.DataFrame(pd.concat((df_1, df_2, df_3, df_4, df_5), axis=0))
# df = df.reset_index(drop=False)
# print(df.columns)
# df = df.drop(['Unnamed: 0'], axis=1)
# df.to_csv('reg_preds_meta.csv')
file_path = 'D:/TalTechUniversity/solarIradiance/forecasting/nZEB_mektory/' \
            'solar_farm/data_prepration/farm/datasets/sf_dataset.csv'
df1 = pd.read_csv(file_path, parse_dates=['date'], index_col=['date', 'time'])

df = df.reindex(df1.index, fill_value=0)
con_df = pd.concat([df1, df], axis=1)
con_df = con_df.drop(['actual'], axis=1)
con_df.to_csv('sf_meta_dt.csv')
# df1 = df1[df1.Power > 0]
# df1.to_csv('sf_dt.csv')
