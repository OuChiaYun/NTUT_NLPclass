import re
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from gensim.models import KeyedVectors
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# 加载數據和模型
df = pd.read_csv('../IMDB/IMDB_Top250Engmovies2_OMDB_Detailed.csv')
df['Genre_lower'] = df['Genre'].apply(lambda x: [i.strip().lower() for i in x.split(',')])
df['Actors_lower'] = df['Actors'].apply(lambda x: [i.strip().lower() for i in x.split(',')][:3])
df['Director_lower'] = df['Director'].apply(lambda x: [i.strip().lower() for i in x.split(',')])
df['Year_str'] = df['Year'].astype(str)
word2vec_model = KeyedVectors.load_word2vec_format('../pre-train-model/GoogleNews-vectors-negative300.bin.gz', binary=True)

def load_crf_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {file_path}")
    return model

crf = load_crf_model('../self-train-model/crf_ner_model.pkl')

def extract_features(sentence):
    tokens = sentence.split()
    features_list = []
    for i in range(len(tokens)):
        word = tokens[i]
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        if i > 0:
            word1 = tokens[i-1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
            })
        else:
            features['BOS'] = True
        
        if i < len(tokens) - 1:
            word1 = tokens[i+1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
            })
        else:
            features['EOS'] = True
        
        features_list.append(features)
    
    return [features_list]

def predict(sentence, crf_model):
    features = extract_features(sentence)
    labels = crf_model.predict(features)
    return list(zip(sentence.split(), labels[0]))

def extract_entities(predicted_labels, entity_type):
    entities = []
    current_entity = []
    
    for word, label in predicted_labels:
        if label == f"B-{entity_type}":
            if current_entity:
                entities.append(" ".join(current_entity))
                current_entity = []
            current_entity.append(word)
        elif label == f"I-{entity_type}" and current_entity:
            current_entity.append(word)
        elif label == "O" and current_entity:
            entities.append(" ".join(current_entity))
            current_entity = []
    
    if current_entity:
        entities.append(" ".join(current_entity))
    
    return entities

def vector_similarity_kinds(word, model):
    return model[word] if word in model else np.zeros(model.vector_size)

def calculate_max_similarity_kinds(search_term, words, model):
    search_terms = search_term.split()
    search_vectors = [vector_similarity_kinds(term, model) for term in search_terms]
    word_vectors = [vector_similarity_kinds(word, model) for word in words]

    max_similarity = 0
    for search_vector in search_vectors:
        if np.linalg.norm(search_vector) == 0:
            continue
        for word_vector in word_vectors:
            if np.linalg.norm(word_vector) == 0:
                continue
            similarity = cosine_similarity([search_vector], [word_vector])[0][0]
            max_similarity = max(max_similarity, similarity)
    
    return max_similarity

def recommend_similar_kinds(type_name, input_kind, df, model):
    type_names = type_name + 's'
    df[type_names] = df[type_name].apply(lambda x: ''.join(x))
    kinds = df[type_names].tolist()
    
    similarities = []
    for k in kinds:
        words = k.split()
        max_similarity = calculate_max_similarity_kinds(input_kind, words, model)
        similarities.append(max_similarity)

    sorted_indices = np.argsort(similarities)[::-1]
    recommended_kinds = set()
    recommended_kind_indices = []
    
    for idx in sorted_indices:
        current_kind = kinds[idx]
        if len(recommended_kind_indices) < 3 or similarities[idx] == 1:
            if current_kind not in recommended_kinds:
                recommended_kinds.add(current_kind)
                recommended_kind_indices.append(idx)

    recommended_movies = []
    for idx in recommended_kind_indices:
        kind = kinds[idx]
        print(f"'{input_kind}' 與 '{kind}' 最相似。 相似度分數: {similarities[idx]:.2f}")
        
        matching_indices = [i for i, k in enumerate(kinds) if k == kind]
        for idx in matching_indices:
            recommended_movies.append([type_name,df['Title'][idx], df['Director'][idx], df['Actors'][idx], df['Genre'][idx], df['Year'][idx], df['Plot'][idx]])
    return recommended_movies

def vector_similarity(word, model):
    return model[word] if word in model else np.zeros(model.vector_size)

def calculate_max_similarity(search_term, words, model):
    search_terms = search_term.split()
    search_vectors = [vector_similarity(term, model) for term in search_terms]
    word_vectors = [vector_similarity(word, model) for word in words]

    max_similarity = 0
    for search_vector in search_vectors:
        if np.linalg.norm(search_vector) == 0:
            continue
        for vec in word_vectors:
            if np.linalg.norm(vec) == 0:
                continue
            similarity = cosine_similarity([search_vector], [vec])[0][0]
            max_similarity = max(max_similarity, similarity)
    
    return max_similarity

def recommend_similar_genres(type_name, input_genre, df, model):
    if type_name == "Year":
        type_name = "Year_str"
        recommended_movies = df[df[type_name] == input_genre][['Title', 'Director', 'Actors', 'Genre', 'Year_str', 'Plot']]
        recommended_movies_list = []
        
        for index, row in recommended_movies.iterrows():
            movie_details = [type_name] + row.tolist()  # 将 type_name 作为列表的第一个元素
            recommended_movies_list.append(movie_details)
        
        return recommended_movies_list

    type_names = type_name + '_s'
    df[type_names] = df['Genre_lower'].apply(lambda x: ' '.join(x))
    genres = df[type_names].tolist()
    
    similarities = []
    for g in genres:
        words = g.split()
        max_similarity = calculate_max_similarity(input_genre, words, model)
        similarities.append(max_similarity)

    sorted_indices = np.argsort(similarities)[::-1]
    recommended_genres = set()
    recommended_genre_indices = []
    max_similarity = similarities[sorted_indices[0]]

    for idx in sorted_indices:
        if len(recommended_genre_indices) < 3 or similarities[idx] == 1:
            current_genre = genres[idx]
            if current_genre not in recommended_genres and current_genre != input_genre:
                recommended_genres.add(current_genre)
                recommended_genre_indices.append(idx)

    recommended_movies = []
    for idx in recommended_genre_indices:
        genre = genres[idx]
        print(f"'{input_genre}' 與 '{genre}' 最相似。 相似度分數: {similarities[idx]:.2f}")
        
        matching_indices = [i for i, g in enumerate(genres) if g == genre]
        for idx in matching_indices:
            recommended_movies.append([type_name,df['Title'][idx], df['Director'][idx], df['Actors'][idx], df['Genre'][idx], df['Year_str'][idx], df['Plot'][idx]])

    return recommended_movies

def handle_search(type_name, word):
    print("type = ", type_name, ", word = ", word)
    if type_name == "Actors" or type_name == "Director":
        results = recommend_similar_kinds(type_name, word, df, word2vec_model)

    else:
        results = recommend_similar_genres(type_name, word, df, word2vec_model)
    
    return results


#################################



cosine_sim_matrix = np.loadtxt('../matrix/sentence_transformers_cosine_similarity_matrix.txt')
indices = pd.Series(df.index, index=df['Title']).drop_duplicates()

def recommend(title, cosine_sim=cosine_sim_matrix):
    recommended_movies = []
    if title in indices:
        idx = indices[title]
        score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
        top_5_indices = list(score_series.iloc[1:6].index)
        
        for i in top_5_indices:
            recommended_movies.append((df['Title'].iloc[i], score_series[i], df['Genre'].iloc[i]))
    return recommended_movies

@app.route('/recommendations/<title>')
def show_recommendations(title):
    recommended_movies = recommend(title)
    movie_details = df[df['Title'].str.lower() == title.lower()].iloc[0]  # 获取电影的详细数据
    return render_template('recommendations.html', movies=recommended_movies, title=title, movie_details=movie_details)


@app.route('/')
def home():
    sorted_df = df.sort_values(by='imdbRating', ascending=True)
    movies = sorted_df.to_dict(orient='records') 
    return render_template('search.html', movies=movies)


@app.route('/search_submit', methods=['POST'])
def search_submit():
    search_query = request.form['search_query']
    search_query = re.sub(r'[^\w\s]', '', search_query)
    predicted_labels = predict(search_query, crf)
    
    actors = extract_entities(predicted_labels, "Actor")
    directors = extract_entities(predicted_labels, "Director")
    years = extract_entities(predicted_labels, "Year")
    genres = extract_entities(predicted_labels, "Genre")

    search_word = []
    for actor in actors:
        search_word.append(['Actors', actor])
    for director in directors:
        search_word.append(['Director', director])
    for year in years:
        search_word.append(['Year', year])
    for genre in genres:
        search_word.append(['Genre', genre])
    
    results = []
    for type_name, word in search_word:
        results.extend(handle_search(type_name, word))

    return render_template('results.html', results=results, query=search_query, actors=actors, directors=directors, years=years, genres=genres)

if __name__ == '__main__':
    app.run(debug=True)
