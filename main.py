import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

movie_dataset = pd.read_csv("/Users/yusragokcecelikel/Documents/GitHub/movie_recommendation_system/imdb_top_1000.csv")
movie_dataset = movie_dataset.iloc[:,1:] #remove unnamed column
print(movie_dataset.columns)
