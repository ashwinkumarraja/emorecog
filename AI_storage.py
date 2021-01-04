#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install silence_tensorflow')
import ga_functions as ga
import model as M
import numpy as np
import pandas as pd
import random
import time

import tensorflow as tf

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


# In[2]:


path = "./dataset.csv"
df = pd.read_csv(path)
df


# In[3]:


df.drop(df.iloc[:, 696:714], inplace=True, axis=1)
df.drop(df.iloc[:, 0:5], inplace=True, axis=1)
N = len(df.columns)
label = df["Label"]
df.drop(columns=['Label'], inplace=True, axis=1)
X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.20, random_state=0)
inital_num_features = len(df.columns)


# In[4]:


#fitness_functions = ["svm", "lr", "nn"]
fitness_functions = ["lr"]
num_generations = [5, 10, 15, 20, 25, 30, 35]
pop_sizes = [50, 75, 100, 125, 150, 175, 200, 225, 250]

selection_functions = ["roulette", "rank", "tournament"]

crossover_functions = ["k_point", "uniform"]
k_values = np.arange(1, 11)

mutations_functions = ["bit_flip", "bit_swap"]
mutations_factors = [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1]

update_functions = ["gen", "weak_p"]


# In[5]:


accuracy_before_GA = M.classify_before_GA_NN(X_train, X_test, y_train, y_test)


# In[6]:


chromo_list = list()
data_list = list()


# In[ ]:


for pop_size in pop_sizes:
    initial_population = ga.initialize_population(pop_size, inital_num_features)
    for num in num_generations:
        for ff in fitness_functions:
            for sf in selection_functions:
                for cf in crossover_functions:
                    for k in k_values:
                        for mf in mutations_functions:
                            for m in mutations_factors:
                                for uf in update_functions:
                                    for i in range(num):
                                        fitness_scores, initial_population = ga.compute_fitness_score(initial_population, X_train, X_test, y_train, y_test)

                                        if sf == "roulette":
                                            improved_population = ga.roulette_wheel_selection(initial_population, fitness_scores)
                                        elif sf == "rank":
                                            improved_population = ga.rank_selection(initial_population, fitness_scores, len(initial_population)//2)
                                        elif sf == "tournament":
                                            improved_population = ga.tournament_selection(initial_population, fitness_scores, num_parents=len(initial_population), tournament_size=4)

                                        if cf == "k_point":
                                            improved_population = ga.k_point_crossover(improved_population, k)
                                        elif cf == "uniform":
                                            improved_population = ga.k_point_crossover(improved_population)

                                        if mf == "bit_flip":
                                            improved_population = ga.bit_flip_mutation(improved_population, m)
                                        elif mf == "bit_swap":
                                            improved_population = ga.bit_flip_mutation(improved_population, m)

                                        if uf == "gen":
                                            new_generation = ga.generational_update(initial_population, improved_population)
                                        elif uf == "weak_p":
                                            new_generation = ga.weak_parents_update(initial_population, fitness_scores, improved_population)

                                        initial_population = new_generation

                                        fitness_scores, initial_population = ga.compute_fitness_score(initial_population, X_train, X_test, y_train, y_test)
                                        indices = np.argsort(fitness_scores)
                                        updated_pop = [initial_population[i] for i in indices][::-1]

                                        accuracy_after_GA = M.fitness_score_NN(updated_pop[0], X_train, X_test, y_train, y_test)
                                        data_list.append([ff, num, pop_size, sf, cf, k, mf, m, uf, accuracy_before_GA, accuracy_after_GA])
                                        chromo_list.append(updated_pop[0])


# In[ ]:


newdf = pd.DataFrame(data_list)


# In[ ]:


chromo_array = np.array(chromo_list)


# In[ ]:


newdf.to_csv('perf_analysis.csv', index = False)


# In[ ]:


from numpy import save
save('chromo_array.npy', chromo_array)


# In[ ]:


from numpy import load
# load array
data = load('chromo_array.npy')
data


# In[ ]:


from google.colab import files
files.download( "chromo_array.npy" )
files.download( "perf_analysis.csv" ) 


# In[ ]:




