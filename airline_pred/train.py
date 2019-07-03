from keras.models import Sequential
from keras.layers import Dense
from prep import get_data, get_xy
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import conf


def get_mlp(inputs=2):
    model = Sequential()

    model.add(Dense(10, input_dim=inputs, activation='relu'))
    model.add(Dense(10))
    model.add(Dense(1))

    return model


def train_model(train_x, train_y, epochs, batches):
    model = get_mlp(train_x.shape[1])

    model.compile(loss='mean_squared_error',
                  optimizer='adam', metrics=['mse', 'mape'])
    model.fit(train_x, train_y, verbose=2,
              epochs=epochs, batch_size=batches)

    return model


def r2_score(y_test, y_pred):
    return 1 - sum((y_test - y_pred) ** 2) / sum((y_test - y_pred.mean()) ** 2)


if __name__ == '__main__':
    data = get_data()
    X, Y, years = get_xy(data, 2)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    epochs, batches = 64, 2
    model = train_model(train_x, train_y, epochs, batches)

    y_pred = np.array(model.predict(test_x)).flatten()
    print(r2_score(test_y, y_pred))

    y_plt = np.array(model.predict(X).flatten())
    plt.plot(years, y_plt)
    plt.plot(years, Y)
