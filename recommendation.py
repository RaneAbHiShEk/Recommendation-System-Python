#backend
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df=pd.read_csv(r'C:\Users\Abhishek\Downloads\movie_dataset.csv')
features=['keywords','cast','genres','director']
def combine_features(row):
    return row['keywords']+" "+row['genres']+" "+row['cast']+" "+row['director']
for feature in features:
    df[feature]=df[feature].fillna('')
df["combined_feature"]=df.apply(combine_features,axis=1)
cv=CountVectorizer()
count_matrix=cv.fit_transform(df['combined_feature'])
cosine_sim=cosine_similarity(count_matrix)
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

con="Top 5 similar movies: \n"
def show(xx):
    movie_user_like=xx
    movie_index=get_index_from_title(movie_user_like)
    similar_movies=list(enumerate(cosine_sim[movie_index]))
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
    i=0
#print("Top 5 similar movies to "+movie_user_like+" are:\n")
    lis1=[]
    for element in sorted_similar_movies:
        print(get_title_from_index(element[0]))
        lis1.append(get_title_from_index(element[0]))
        i=i+1
        if i>5:
            break
    global con
    for x in lis1:
        #con=global con
        con=con+x+"\n"