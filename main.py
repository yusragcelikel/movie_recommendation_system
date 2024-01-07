import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn import preprocessing
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity




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
def preprocess(df):
    #combine all text columns
    s = list(df.select_dtypes(include=['object']).columns) #store all object data type in list
    s.remove('Series_Title') # Remove Title column
    df['all_text'] = df[s].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1) # Joining all object columns using commas into a column

    # Create a tokenizer
    token = RegexpTokenizer(r'[a-zA-Z]+')

    # Converting TfidfVector from the text
    cnvrt_text = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,1), tokenizer=token.tokenize)
    text_counts = cnvrt_text.fit_transform(df['all_text'])

    # Select numerical variables
    numerical_df = df.select_dtypes(include=['float64', 'int64'])

    # Scaling Numerical variables
    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

    # Applying scaler on our data and converting it into a dataframe
    mx_numerical_df = pd.DataFrame((scaler.fit_transform(numerical_df)))
    mx_numerical_df.columns = numerical_df.columns

    # Adding numerical variables in the TF-IDF vector
    IMDB_Rating = mx_numerical_df.IMDB_Rating.values[:, None]
    X_train_dtm = hstack((text_counts, IMDB_Rating))

    return X_train_dtm


# preprocessing the data
sparse_matrix = preprocess(movie_dataset_genres)
#print(sparse_matrix.shape)

# Compute the sigmoid kernel
sigmoid_kernel = cosine_similarity(sparse_matrix, sparse_matrix)

# Reverse mapping of indices and movie titles
indices = pd.Series(movie_dataset_genres.index, index=movie_dataset_genres['Series_Title']).drop_duplicates()
#print(indices.head(10))


# Writing a function to get recommendations based on the similarity score
def give_recommendation(title, sig=sigmoid_kernel):
    idx = indices[title] # Get the index corresponding to original_title
    sig_scores = list(enumerate(sig[idx])) # Get the pairwise similarity scores
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True) # Sort the movies
    sig_scores = sig_scores[1:11] # Scores of the 10 most similar movies
    movie_indices = [i[0] for i in sig_scores] # Movie indices
    top_10_most_similar = movie_dataset_genres['Series_Title'].iloc[movie_indices]  # Top 10 most similar movies
    return top_10_most_similar

print(give_recommendation("The Matrix", sig = sigmoid_kernel))

