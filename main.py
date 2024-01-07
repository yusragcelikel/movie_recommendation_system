import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

movie_dataset = pd.read_csv("/Users/yusragokcecelikel/Documents/GitHub/movie_recommendation_system/imdb_top_1000.csv")
movie_dataset = movie_dataset.iloc[:,1:] #remove Poster_Link column
#print(movie_dataset.head())
#print(movie_dataset.info())


#print(movie_dataset.Certificate.unique()) #checking unique certificate types

#print(movie_dataset['Genre'].head())

#handling multiple genres given in a single cell of dataframe
genres = movie_dataset['Genre'].str.split(',').apply(pd.Series, 1).stack() #split the values by comma and stack them one after the other
#print(genres.head(20))

genres.index = genres.index.droplevel(-1) #to drop the last level of the index
#print(genres.head(20))

genres.name = 'Genre' # Assign name to column
del movie_dataset['Genre'] # delete column 'Genre' from original dataframe
movie_dataset_genres = movie_dataset.join(genres) #stacked Series is joined with the original dataframe
print(movie_dataset_genres.head(8))


