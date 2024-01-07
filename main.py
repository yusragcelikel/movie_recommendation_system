import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn import preprocessing




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
#print(movie_dataset_genres.head(8))
print(movie_dataset_genres.info())



# Count of movies according to genre using seaborn
#plt.figure(figsize=(15,5))
#sns.countplot(x='Genre', data=movie_dataset_genres)
#plt.xticks(rotation=90)
#plt.tight_layout() #to adjust the figure margins to ensure that all titles fit within the plot
#plt.show()



#the function for preprocessing
def preprocess(movie_dataset_genres):
    #combine all text columns
    s = list(movie_dataset_genres.select_dtypes(include=['object']).columns) #store all object data type in list
    s.remove('Series_Title') # Remove Title column
    movie_dataset_genres['all_text'] = movie_dataset_genres[s].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1) # Joining all object columns using commas into a column

    # Create a tokenizer
    token = RegexpTokenizer(r'[a-zA-Z]+')

    # Converting TfidfVector from the text
    cnvrt_text = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,1), tokenizer=token.tokenize)
    text_counts = cnvrt_text.fit_transform(movie_dataset_genres['all_text'])

    # Select numerical variables
    numerical_movie_dataset_genres = movie_dataset_genres.select_dtypes(include=['float64', 'int64'])

    # Scaling Numerical variables
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

    # Applying scaler on our data and converting it into a dataframe
    mx_numerical_movie_dataset_genres = pd.DataFrame((scaler.fit_transform(numerical_movie_dataset_genres)))
    mx_numerical_movie_dataset_genres.columns = numerical_movie_dataset_genres.columns




