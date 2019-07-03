import pandas as pd
import collections
import numpy as np
import matplotlib.pyplot as plt


airline_data = pd.read_csv('data/airline_data.csv', header=2)


def get_data(col_name='Country Name', value='World'):
    sel = airline_data.loc[airline_data[col_name] == value].iloc[:, 18:-1]
    return [int(k) for k in list(sel)],  sel.values[0]


def get_xy(data, last_values=1):
    years, vol = data
    x, y = [], []
    curr = collections.deque(vol[:last_values])
    for i in range(last_values, len(vol)):
        x.append(list(curr))
        y.append(vol[i])
        curr.append(vol[i])
        curr.popleft()

    return (np.array(x), np.array(y), years[last_values:])


if __name__ == '__main__':
    d = get_data()
    plt.plot(d[0], d[1])
