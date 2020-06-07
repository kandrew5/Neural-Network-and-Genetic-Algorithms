import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
from keras.regularizers import l1

movie_list = []
# append each row in a list
with open('u.txt', newline = '') as movie:
    movie_reader = csv.reader(movie, delimiter='\t')
    for movie in movie_reader:
        movie_list.append(movie)

# Create a dataframe from the list
TrainFrame = pd.DataFrame(movie_list, columns = ['userid', 'movieid', 'rating', 'timestamp'])
TrainFrame = TrainFrame.apply(pd.to_numeric)

# get all distinct users
users = TrainFrame.userid.unique()
mean = {}
# Centering
for user in users:
    mean[user] = TrainFrame.loc[TrainFrame['userid']== user, 'rating'].mean()
    TrainFrame.loc[TrainFrame['userid'] == user, 'rating'] -= mean[user]

    max_val = TrainFrame.rating.max()  # min value
    min_val = TrainFrame.rating.min()  # max value
# create a dataframe with all users and all the rating (even the null ones)
user_movie = pd.pivot_table(TrainFrame, index='userid', columns= 'movieid', values='rating')

# rescaling
for user in users:
    new_mean =  user_movie.loc[user].mean(skipna=True)
    user_movie.loc[user].fillna(new_mean, inplace=True)

column_maxes = user_movie.max()
df_max = column_maxes.max()
column_mins = user_movie.min()
df_min = column_mins.min()
user_movie = (user_movie - df_min) / (df_max - df_min)
user_movie = user_movie.reset_index()

x = pd.get_dummies(user_movie.userid)
y = user_movie.loc[:, user_movie.columns != 'userid']


def training_network():
    kfold = KFold(n_splits=5, shuffle=True)
    rmseList = []
    maeList = []

    i = 0
    for train_index, test_index in kfold.split(x):
        model = keras.Sequential()
        model.add(Dense(943, activation="sigmoid", input_dim=943))
        model.add(Dense(7, activation="relu"))
        model.add(Dense(1682, activation="sigmoid"))

        # Compile model
        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

        def mae(y_true, y_pred):
            return K.mean(abs(y_pred - y_true))

        keras.optimizers.SGD(lr=0.1, momentum=0.6, nesterov=False)
        model.compile(loss='mean_squared_error', optimizer='sgd', metrics=[rmse, mae])

        # simple early stopping
        es = EarlyStopping(monitor='loss', mode='min', verbose=0)
        # Fit model
        history = model.fit(x.iloc[train_index], y.iloc[train_index],
                            validation_data=(x.iloc[test_index], y.iloc[test_index]), epochs=100, batch_size=5,
                            callbacks=[es], verbose=1)

        # Evaluate model
        scores = model.evaluate(x.iloc[test_index], y.iloc[test_index], verbose=0)
        rmseList.append(scores[1])
        maeList.append(scores[2])
        print("Fold :", i, " RMSE:", scores[1])
        i += 1

    print("RMSE: ", np.mean(rmseList))
    print("ΜΑΕ: ", np.mean(maeList))

    pyplot.title('Mean Squared Error')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()


def training_regularize():
    kfold = KFold(n_splits=5, shuffle=True)
    rmseList = []
    maeList = []

    i = 0
    for train_index, test_index in kfold.split(x):
        model = keras.Sequential()
        model.add(Dense(943, activation="sigmoid", input_dim=943))
        model.add(Dense(7, activation="relu", kernel_regularizer=l1(0.9), bias_regularizer=l1(0.9)))
        model.add(Dense(1682, activation="sigmoid"))

        # Compile model
        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

        def mae(y_true, y_pred):
            return (K.mean(abs(y_pred - y_true)))

        keras.optimizers.SGD(lr=0.1, decay=0.0, momentum=0.6, nesterov=False)
        model.compile(loss='mean_squared_error', optimizer='sgd', metrics=[rmse, mae])

        # simple early stopping
        es = EarlyStopping(monitor='loss', mode='min', verbose=1)
        # Fit model
        history2 = model.fit(x.iloc[train_index], y.iloc[train_index],
                             validation_data=(x.iloc[test_index], y.iloc[test_index]), epochs=100, batch_size=5,
                             callbacks=[es], verbose=0)

        # Evaluate model
        scores2 = model.evaluate(x.iloc[test_index], y.iloc[test_index], verbose=0)
        rmseList.append(scores2[1])
        maeList.append(scores2[2])
        print("Fold :", i, " RMSE:", scores2[1])
        i += 1

    print("RMSE: ", np.mean(rmseList))
    print("ΜΑΕ: ", np.mean(maeList))


def training_deep():
    kfold = KFold(n_splits=5, shuffle=True)
    rmseList = []
    maeList = []

    i = 0
    for train_index, test_index in kfold.split(x):
        model = keras.Sequential()
        model.add(Dense(943, activation="sigmoid", input_dim=943))
        model.add(Dense(7, activation="relu"))
        model.add(Dense(7, activation="relu"))
        model.add(Dense(7, activation="relu"))
        model.add(Dense(1682, activation="sigmoid"))

        # Compile model
        def rmse(y_true, y_pred):
            return K.sqrt(K.mean(K.square(y_pred - y_true)))

        def mae(y_true, y_pred):
            return (K.mean(abs(y_pred - y_true)))

        keras.optimizers.SGD(lr=0.1, decay=0.0, momentum=0.6, nesterov=False)
        model.compile(loss='mean_squared_error', optimizer='sgd', metrics=[rmse, mae])

        # simple early stopping
        es = EarlyStopping(monitor='loss', mode='min', verbose=1)
        # Fit model
        history2 = model.fit(x.iloc[train_index], y.iloc[train_index],
                             validation_data=(x.iloc[test_index], y.iloc[test_index]), epochs=100, batch_size=10,
                             callbacks=[es], verbose=0)

        # Evaluate model
        scores1 = model.evaluate(x.iloc[test_index], y.iloc[test_index], verbose=0)
        rmseList.append(scores1[1])
        maeList.append(scores1[2])
        print("Fold :", i, " RMSE:", scores1[1])
        i += 1

    print("RMSE: ", np.mean(rmseList))
    print("ΜΑΕ: ", np.mean(maeList))

