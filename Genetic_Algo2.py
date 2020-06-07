from random import randrange, random
from tkinter import StringVar

import pandas as pd
import numpy as np
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from matplotlib import pyplot as plt

pd.options.mode.chained_assignment = None
import pyeasyga

movie_list = []
with open('u.txt', newline='') as movie:
    movie_reader = csv.reader(movie, delimiter='\t')
    for movie in movie_reader:
        movie_list.append(movie)
TrainFrame = pd.DataFrame(movie_list, columns=['userid', 'movieid', 'rating', 'timestamp'])
TrainFrame = TrainFrame.apply(pd.to_numeric)

users = TrainFrame.userid.unique()  # get unique users

# create a dataframe with all users and all the rating (even the null ones)
user_movie_nan = pd.pivot_table(TrainFrame, index='userid', columns='movieid', values='rating')
user_movie = pd.pivot_table(TrainFrame, index='userid', columns='movieid', values='rating')

for user in users:
    new_mean = round(user_movie.loc[user].mean(skipna=True))
    user_movie.loc[user].fillna(new_mean, inplace=True)

user_ga = 0 #USER

# calculate pearson correlation to find neighbours
PearsonCorr = user_movie.T.corr(method='pearson')
neighbour = PearsonCorr.nlargest(11, 1).index.tolist()  # get the 10 closest neighbours
neighbour = [x - 1 for x in neighbour]  # iloc index starts at 0

Neighbframe = user_movie.iloc[neighbour]
# get all movies without rating
nan_columns = user_movie_nan.columns[user_movie_nan.iloc[user_ga].apply(np.isnan)]
nan_columns = [x - 1 for x in nan_columns]  # iloc index starts at 0
nan_columns = list(range(nan_columns[0] - 10, nan_columns[0])) + nan_columns  # HOLDOUT
data_movie_nan = user_movie_nan.iloc[user_ga, nan_columns].array

# initialize genetic algorithm with the specified parameters
ga = pyeasyga.GeneticAlgorithm(data_movie_nan,
                               population_size=200,
                               generations=300,
                               crossover_probability=0.9,
                               mutation_probability=0.01,
                               elitism=True,
                               maximise_fitness=True)


# create an individual for the starting population
def create_individual(data):
    individual = np.random.randint(1, 6, data.shape)
    return individual


ga.create_individual = create_individual


# define crossover operator for the genetic algorithm (single point)
def crossover(parent_1, parent_2):
    if type(parent_1) is np.ndarray:
        parent_1 = parent_1.tolist()
    if type(parent_2) is np.ndarray:
        parent_2 = parent_2.tolist()
    index = randrange(1, len(parent_1))
    child_1 = parent_1[:index] + parent_2[index:]
    child_2 = parent_2[:index] + parent_1[index:]
    return child_1, child_2


ga.crossover_function = crossover


# define and set the GA's mutation operation
def mutate(individual):
    # change a random gene in the chromosome
    mutate_index = randrange(len(individual))
    individual[mutate_index] = randrange(1, 6)


ga.mutate_function = mutate

# define and set the GA's selection operation   (tournament selection)

ga.selection_function = ga.tournament_selection


# define a fitness function
def fitness(individual, data):
    pearson_corr = 0
    for no, index in zip(individual, nan_columns):
        Neighbframe.iloc[0, index] = int(no)
    PearsonCorr = Neighbframe.T.corr(method='pearson')
    pearson_corr = ((PearsonCorr.sum().iloc[0] - 1) / 10) + 1
    return pearson_corr


ga.fitness_function = fitness  # set the GA's fitness function
temp_chrom, holdout = ga.run()  # run the GA
best_chrom_mean = temp_chrom
y_actual = user_movie.iloc[0, 0:10].tolist()
rmse = []
mae = []
t_rmse = []
t_mae = []

for genes in holdout:
    genes = genes[:10]
    rmse.append(sqrt(mean_squared_error(y_actual, genes)))
    mae.append((mean_absolute_error(y_actual, genes)))
print(ga.best_individual())
for i in range(9):
    temp_chrom, holdout = ga.run()
    t_rmse = []
    t_mae = []
    for genes in holdout:
        genes = genes[:10]
        t_rmse.append(sqrt(mean_squared_error(y_actual, genes)))
        t_mae.append((mean_absolute_error(y_actual, genes)))
    rmse = [x + y for x, y in zip(rmse, t_rmse)]
    mae = [x + y for x, y in zip(mae, t_mae)]

    if len(temp_chrom) < len(best_chrom_mean):
        temp_chrom.extend([ga.best_individual()[0]] * (len(best_chrom_mean) - len(temp_chrom)))
    best_chrom_mean = [x + y for x, y in zip(best_chrom_mean, temp_chrom)]
    print(ga.best_individual())
best_chrom_mean = [number / 10 for number in best_chrom_mean]
rmse = [number / 10 for number in rmse]
mae = [number / 10 for number in mae]
print(best_chrom_mean)
plt.plot(best_chrom_mean)
plt.show()

plt.plot(rmse)
plt.show()

plt.plot(mae)
plt.show()
print(rmse)
print(mae)
