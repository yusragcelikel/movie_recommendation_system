# Movie Recommendation System
This project developed using Python and various libraries including numpy, pandas, matplotlib, seaborn, and scikit-learn. Pandas was used for data preprocessing. 
Text data was vectorized using TfidfVectorizer, and numerical data was scaled using MinMaxScaler. 
Text data vectorized with TfidfVectorizer was combined with numerical data to create a similarity matrix using cosine_similarity. 
Movie recommendations were obtained based on the movie title entered by the user, by determining the 10 closest movies in the similarity matrix.
