# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 14:04:38 2017

@author: gkalra
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from collections import Counter
from sklearn.neighbors import KDTree

data = pd.read_csv(<path to file>)
#data.info()

first_actors = set(data.actor_1_name.unique())
second_actors = set(data.actor_2_name.unique())
third_actors = set(data.actor_3_name.unique())
#print('Those only in first name', len(first_actors - second_actors - third_actors))
#print('Those only in second name', len(second_actors - first_actors - third_actors))
#print('Those only in third name', len(third_actors - first_actors - second_actors))

data.color = data.color.map({'Color' : 1, 'Black and White' : 0})
#print(data.color.head())

unique_genres = set()
for genre_labels in data.genres.str.split('|').values:
    unique_genres = unique_genres.union(set(genre_labels))
#print(unique_genres)
for genre in unique_genres:
    data["genre_" + genre] = data.genres.str.contains(genre).astype(int)
data = data.drop('genres', axis = 1)

if len(data.drop_duplicates(subset = ['movie_title', 'title_year', 'movie_imdb_link'])) < len(data):
    print('Duplicates found')
    #duplicates = data[data.movie_title.map(data.movie_title.value_counts() > 1)]
    #print(duplicates.info())
    print('Removing duplicates')
    data = data.drop_duplicates(subset = ['movie_title', 'title_year', 'movie_imdb_link'])
    #data.info()

data.language = data.language.map(data.language.value_counts())
#print(data.language)
data.country = data.country.map(data.country.value_counts())
#print(data.country)
data.content_rating = data.content_rating.map(data.content_rating.value_counts())
#print(data.content_rating)
data.director_name = data.director_name.map(data.director_name.value_counts()) 

unique_words = set()
for wordlist in data.plot_keywords.str.split('|').values:
    if wordlist is not np.nan:
        unique_words = unique_words.union(set(wordlist))
plot_wordlist = list(unique_words)
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')
for word in plot_wordlist:
    data['plot_has_' + word.replace(' ', '-')] = data.plot_keywords.str.contains(word).astype(float)
data = data.drop('plot_keywords', axis=1)

#print(data.select_dtypes(include = ['O']).columns)
data = data.drop('movie_imdb_link', axis = 1)

actor_count = pd.concat([data.actor_1_name, data.actor_2_name, data.actor_3_name]).value_counts()
data.actor_1_name = data.actor_1_name.map(actor_count)
data.actor_2_name = data.actor_2_name.map(actor_count)
data.actor_3_name = data.actor_3_name.map(actor_count)

#print(data.select_dtypes(include = ['O']).columns)
#print (data.isnull().sum(axis = 1))

df = data.dropna(thresh = 100)

#print (df.isnull().sum(axis = 1))

def fill_missing(data, column, classifier):
    df = data.dropna(subset=[col for col in data.columns if col != column])
    nullmask = df[column].isnull()
    train, test = df[~nullmask], df[nullmask]
    train_x, train_y = train.drop(column, axis = 1), train[column]
    classifier.fit(train_x, train_y)
    if len(test) > 0:
        test_x = test.drop(column, axis = 1)
        test_y = classifier.predict(test_x)
        new_x, new_y = pd.concat(train_x, test_x), pd.concat(train_y, test_y)
        newdf = new_x[column] = new_y
        return newdf
    else:
        return df
    
c, r = KNeighborsClassifier, KNeighborsRegressor
title_encoder = LabelEncoder()
title_encoder.fit(df.movie_title)
df.movie_title = title_encoder.transform(df.movie_title)

#print(df[df.columns[:25]].isnull().sum())

imputation_order = [('director_name', c), ('title_year', c),
                ('actor_1_name', c), ('actor_2_name', c), ('actor_3_name', c),
                ('gross', r), ('budget', r), ('aspect_ratio', r),
                ('content_rating', r), ('num_critic_for_reviews', r)]
for column, classifier in imputation_order:
    df = fill_missing(df, column, classifier())
    print("column " + column + ": imputation complete")

#print(df[df.columns[:25]].isnull().sum())

movie_titles = title_encoder.inverse_transform(df.movie_title)

def get_movie_codes(names):
    movies = []
    for name in names:
        movie = [i for i in movie_titles if name.lower() in i.lower()]
        if len(movie) > 0:
            movies.append(movie[0])
            print("added movie " + movie[0] + " to movies") 
    
    movie_codes = title_encoder.transform(movies)
    return movies, movie_codes

def recommend(data, movies, titles, tree):
    titles = list(titles)
    length = len(movies) + 1
    rec = []
    for i, movie in enumerate(movies):
        weight = length - i
        distance, indices = tree.query(data[titles.index(movie)].reshape(1, -1), k = 3)
        for d, m in zip(distance[0], indices[0]):
            rec.append((d * weight, titles[m]))
    recommendations = [j[1].strip() for j in rec if j[1] not in movies]
    recommendations = [j[1] for j in sorted([(v, k) for k, v in Counter(recommendations).items()], reverse = True)]                       
    return recommendations

	
ndf = df.drop('movie_title', axis = 1)
ndf = MinMaxScaler().fit_transform(ndf)    
names = ['avatar', 'Harry Potter and the Half-Blood Prince', 'inside out', 'inception', 'Ratatouille']
movies, movie_codes = get_movie_codes(names)
kd_tree = KDTree(ndf, leaf_size = 2)
recs = recommend(ndf, movies, movie_titles, kd_tree)
print("Recommended for you: ")
fmt = '{}.   | {}'
for index, movie in enumerate(recs):
    print(fmt.format(index + 1, movie))