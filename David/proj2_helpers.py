
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_avg_rating(movies_df, data_arr):
    avg = np.zeros(len(movies_df))
    for movie_id in movies_df['Movie Id']:
        inds = (data_arr[:,1] == movie_id)
        avg[movie_id-1] = (np.mean(data_arr[:,2][inds]))
    movies_df['Avg'] = avg

def get_num_rating(movies_df, data_arr):
    movies_df['Num'] = np.bincount(data_arr[:,1])[1:]
    
def get_title_from_id(movies, movie_id):
    '''Returns the movie title corresponding to the movie id'''
    return movies.iloc[movie_id-1]['Movie Title']

def data_from_ids(movies_df, movie_ids, data_arr):
    '''
    Returns the subset of data_arr corresponding to the movie ids
    
    Inputs:
        movies_df: pandas dataframe of movies. Each row represents one movie, 
        and contains information on the genre, movie id, and movie title,
        and average rating
        
        movie_ids: an int or an iterable of ints with the movie ids to include
        
        data_arr: numpy array of all rating data, each row contains
        the user id, movie id, and a rating from 1 to 5
        
    Returns:
        An numpy array of the ratings for the data matching the inputted ids
        
    '''
    inds = np.zeros(len(data_arr), dtype=bool)
    if type(movie_ids) == int or type(movie_ids) == np.int64:    
        inds = (data_arr[:,1] == movie_ids)
    else:
        for movie_id in movie_ids:
            inds += (data_arr[:,1] == movie_id)
    return data_arr[inds]

def hist_from_data(movies_df, data_arr, plot_fig=True, save_fig=False, title=''):
    '''
    Plots the histogram of movie distributions for a single movie.
    
    Inputs:
        movies_df: pandas dataframe of movies. Each row represents one movie, 
        and contains information on the genre, movie id, and movie title,
        and average rating
        
        data_arr: a numpy array of the rating data, containing in each row a
        user id, a movie id, and a rating from 1 to 5.
        
        plot_fig: boolean variable for whether of plot the histogram (default == True)
    
        save_plot: boolean variable for whether to save the histogram. (default == False)
    
    Returns:
        The numpy arrays of labels and frequencies used to plot the histogram
    
    Also prints the histogram to output
    '''                            
    freq, labels = np.histogram(data_arr[:,2], bins = np.arange(6)+.5)
    labels = (labels[1:] + labels[:-1])/2
    plt.figure()
    plt.title('Distribution of Ratings of {}'.format(title))
    plt.bar(labels, freq)
    plt.xlabel('Rating')
    plt.ylabel('Number')
    if plot_fig == True:
        plt.show()
    if save_fig == True:
        plt.savefig(title)
    return labels, freq

def get_genre_ids(movies_df, genre, n=-1):
    '''Gets the first n movie ids of the given genre
    
    Inputs:
        movies_df: pandas dataframe with movie informations, including
        title, average rating, and genre information.
        
        genre: the genre to retrieve the ids for
        
        n: the number of ids to retrieve
        
    Output:
        a numpy array of integers corresponding to the id of the movies
    '''
    return movies_df[movies_df[genre]==1]['Movie Id'][:n]


def plot_proj(proj, movies_df, movie_ids, id_label=None, label_pts=True, suppress=False, box_color='yellow', size=(10, 10)):
    '''
    Plots the movies given by the ids on the 2D projection
    
    Inputs:
        proj: the 2D projection of the factorized "V" matrix. 
        
        movies_df: pandas dataframe with movie informations, including
        title, average rating, and genre information.
        
        movie_ids: numpy array of integers corresponding to the movies to plot.
        
        id_label: the label of the points in the legend. Relevant for multiple 
        datasets. (default = None)
        
        label_pts: boolean of whether to label plotted point with movie titles
        (default = True)
        
        suppress: boolean of whether to initialize and display the plot. To 
        plot multiple plots on the same figure, let suppress=True. (default = False)
        
        box_color: color of shading of the labels (default = yellow)
        
        size: size of the plot
        
    Outputs:
        None.
    '''
    if suppress == False:
        plt.figure(figsize=size)
    
    mask = np.zeros(len(proj[0]), dtype=bool)
    mask[movie_ids-1] = True
    
    x = proj[0][mask]
    y = proj[1][mask]
    
    plt.scatter(x, y, label=id_label)
    if label_pts == True:
        labels = [get_title_from_id(movies, movie_id) for movie_id in movie_ids]
        for label, x1, y1 in zip(labels, x, y):
            plt.annotate(
                label,
                xy=(x1, y1), xytext=(-5, 5),
                textcoords='offset points', ha='right', va='bottom'
                ,bbox=dict(boxstyle='round,pad=0.5', fc=box_color, alpha=0.5)
                #,arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0')
            )
    plt.axis('equal')
    if suppress == False:
        plt.show()

if __name__ == "__main__":
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
        
        



