# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:43:44 2018

@author: David Wang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from proj2_helpers import *

data = np.loadtxt('data/data.txt', dtype=int)
movies = pd.read_csv('data/movies.csv', sep=',', encoding='latin1')

# Plot the distribution of all ratings
_ = hist_from_data(movies, data, title='All Ratings')

# Plot ratings for 10 most rated movies
movie_id_sorted = np.bincount(data[:,0])[::-1]
top_10_ratings = movie_id_sorted[:10]

for movie_id in top_10_ratings:
    movie_data = data_from_ids(movies, movie_id, data)
    movie_title = get_title_from_id(movies, movie_id)
    hist_from_data(movies, movie_data, title=movie_title)

# Plot Ratings for 10 highest rated movies
get_avg_rating(movies, data)

top_10_highest = (movies.sort_values('Avg', ascending=False)[:10]['Movie Id'])

for movie_id in top_10_highest:
    movie_data = data_from_ids(movies, movie_id, data)
    movie_title = get_title_from_id(movies, movie_id)
    hist_from_data(movies, movie_data, title=movie_title)

# Plot ratings for the following genres
genres = ['Action', 'Comedy', 'Film-Noir']

for genre in genres:
    ids = movies[movies[genre]==1]['Movie Id']
    movie_data = data_from_ids(movies, ids, data)
    movie_title = genre+' Movies'
    hist_from_data(movies, movie_data, title=movie_title)


