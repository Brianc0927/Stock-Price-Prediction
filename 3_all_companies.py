from sklearn.linear_model import LinearRegression
import pandas_datareader as web
import numpy as np
import datetime

df_tsla = web.DataReader(
    'TSLA',
    data_source='yahoo',
    start='2022-01-01',
    end='2022-07-09'
)
time_list = df_tsla.index.tolist()
time_list = [ time.to_pydatetime().strftime('%Y-%m-%d') for time in time_list ]
# print(time_list)
# input()
df_output = df_tsla['Close'].to_numpy()

data_closing_dict = dict()

# TSLA
data_closing_dict['TSLA'] = df_output

name_list = ['TSLA', 'RIVN', 'LCID', 'XPEV', 'LI', 'PTRA', 'F', 'GM', 'TWTR', 'TGT', 'SVNDY', 'EMR', 'GRMN', 'DHR', 'NUE', 'NSANY', 'TM', 'HMC']

def get_r2_numpy(predictions, targets):
    return np.corrcoef(predictions, targets)[0, 1]**2

for item in name_list:
    df_input = web.DataReader(
        item,
        data_source='yahoo',
        start='2022-01-01',
        end='2022-07-09'
    )
    data_closing_dict[item] = df_input['Close'].to_numpy()
    df_input = df_input['Close'].to_numpy()[:,None]

    minor_ols = LinearRegression()
    minor_ols.fit(df_input, df_output[:,None])
    predicted_y = minor_ols.predict(df_input)
    # print(predicted_y.shape)
    predicted_y = np.squeeze(predicted_y, axis=1)

    print("{}: {}".format(item, get_r2_numpy(df_output,predicted_y)))

# save train data
with open('data.csv', 'w') as f:
    f.write("{}\n".format("\t".join(['Date']+name_list)))
    for i, time in enumerate(time_list):
        line_list = [time]
        for item in name_list:
            line_list.append(str(data_closing_dict[item][i]))
        f.write("{}\n".format("\t".join(line_list)))