# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:12:08 2017

@author: jens
"""

# Importing Modules
import pandas as pd  
import numpy as np  

# Importing Data
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

movies_df.columns = ['movieId', 'movie_title','genres']

# We dont need timestamp of rating
del ratings_df['timestamp']

# Replacing movieId with movie_title for easier understanding
ratings_df = pd.merge(ratings_df, movies_df, on='movieId')[['userId', 'movie_title','rating', 'movieId', 'genres']]

# Convert ratings_df to matrix with movies as columns and users as rows
ratings_mtx_df = ratings_df.pivot_table(values='rating', index='userId', columns='movie_title')  
ratings_mtx_df.fillna(0, inplace=True)
movie_index = ratings_mtx_df.columns

# Make a correlation matrix to measure correlation between two variables
corr_matrix = np.corrcoef(ratings_mtx_df.T)  


"""# Converting movie titles to dummy variables (for later use)
movies_df = pd.concat([movies_df, movies_df.genres.str.get_dummies(sep='|')], axis=1)  
movies_df.head()

movie_categories = movies_df.columns[3:]  
movies_df.loc[0]"""

# Take list of user ratings and correlate them with all other ratings to return list of recommended movies
def get_movie_similarity(movie_title):  
    '''Returns correlation vector for a movie'''
    movie_idx = list(movie_index).index(movie_title)
    return corr_matrix[movie_idx]

def get_movie_recommendations(user_movies):  
    '''given a set of movies, it returns all the movies sorted by their correlation with the user'''
    movie_similarities = np.zeros(corr_matrix.shape[0])
    for movie_id in user_movies:
        movie_similarities = movie_similarities + get_movie_similarity(movie_id)
    similarities_df = pd.DataFrame({
        'movie_title': movie_index,
        'sum_similarity': movie_similarities
        })
    similarities_df = similarities_df[~(similarities_df.movie_title.isin(user_movies))]
    similarities_df = similarities_df.sort_values(by=['sum_similarity'], ascending=False)
    return similarities_df

# Select user and view their ratings
sample_user = 672  
ratings_df[ratings_df.userId==sample_user].sort_values(by=['rating'], ascending=False)

# Recommend new movies based on previous ratings
sample_user_movies = ratings_df[ratings_df.userId==sample_user].movie_title.tolist()  
recommendations = get_movie_recommendations(sample_user_movies)
recommendations = pd.merge(recommendations, movies_df, on= 'movie_title')[['movie_title', 'movieId', 'genres']]

#We get the top 20 recommended movies
recommendations.head(20)