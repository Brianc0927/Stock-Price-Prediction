from sklearn.linear_model import LinearRegression
import pandas_datareader as web
import numpy as np

df_rivn = web.DataReader(
    'RIVN',
    data_source='yahoo',
    start='2022-01-01',
    end='2022-07-09'
)

df_tsla = web.DataReader(
    'TSLA',
    data_source='yahoo',
    start='2022-01-01',
    end='2022-07-09'
)

df_input = df_rivn['Close'].to_numpy()[:,None]
df_output = df_tsla['Close'].to_numpy()[:,None]

def get_r2_numpy(predictions, targets):
    return np.corrcoef(predictions, targets)[0, 1]**2

minor_ols = LinearRegression()
minor_ols.fit(df_input, df_output)
predicted_y = minor_ols.predict(df_input)
predicted_y = np.squeeze(predicted_y, axis=1)
df_output = np.squeeze(df_output, axis=1)

print(get_r2_numpy(df_output,predicted_y))