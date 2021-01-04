"""
This file has the main code to run the GA for feature select and feed it to ML Model.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

import ga_functions as ga
import model as M


path = "dataset.csv"
df = pd.read_csv(path).sample(frac=1)
df.drop(df.iloc[:, 696:714], inplace=True, axis=1)
df.drop(df.iloc[:, 0:5], inplace=True, axis=1)
N = len(df.columns)
label = df["Label"]
df.drop(columns=['Label'], inplace=True, axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    df, label, test_size=0.20, random_state=0)

# print("X_train =", str(X_train.shape))
# print("X_test =", str(X_test.shape))
# print("y_train =", str(y_train.shape))
# print("y_test =", str(y_test.shape))

# Genetic Algorithm Hyperparameters:
NUM_GENERATIONS = 5
INIT_POP_SIZE = 10  # higher the better
BEST_K_CHROM = INIT_POP_SIZE // 2
CROSSOVER_RATIO = 1
MUTATION_FACTOR = 0.2
NUM_FEATURES = len(df.columns)
THRESHOLD = 0.05

# Suppport Vector Machine Hyperparameters

# Neural Network Hyperparameters
EPOCHS = 250


if __name__ == "__main__":
    accuracy_before_GA = M.classify_before_GA_NN(
        X_train, X_test, y_train, y_test)
    print("ACCURACY FOR MODEL WITHOUT GA =", accuracy_before_GA)

    initial_population = ga.initialize_population(INIT_POP_SIZE, NUM_FEATURES)

    for i in range(NUM_GENERATIONS):
        print("----GENERATION #" + str(i+1) + "----")
        fitness_scores, initial_population = ga.compute_fitness_score(
            initial_population, X_train, X_test, y_train, y_test)
        improved_population = ga.roulette_wheel_selection(
            initial_population, fitness_scores)
        # improved_population = ga.tournament_selection(initial_population, fitness_scores, num_parents = 20, tournament_size = 4)
        # improved_population = ga.rank_selection(initial_population, fitness_scores, BEST_K_CHROM)
        # cross_overed_pop = ga.uniform_crossover(improved_population)
        cross_overed_pop = ga.k_point_crossover(
            improved_population, CROSSOVER_RATIO)
        mutated_pop = ga.bit_flip_mutation(cross_overed_pop, MUTATION_FACTOR)
        new_generation = ga.weak_parents_update(
            initial_population, fitness_scores, mutated_pop)
        initial_population = new_generation

    print("----FINAL CALCULATIONS----")
    fitness_scores, initial_population = ga.compute_fitness_score(
        initial_population, X_train, X_test, y_train, y_test)
    indices = np.argsort(fitness_scores)
    updated_pop = [initial_population[i] for i in indices][::-1]

    accuracy_after_GA = M.fitness_score_NN(
        updated_pop[0], X_train, X_test, y_train, y_test)
    print("ACCURACY OF MODEL AFTER GA =", accuracy_after_GA)
