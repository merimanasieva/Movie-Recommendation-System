import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = {
    'title': [
        'The Matrix',
        'The Godfather',
        'The Dark Knight',
        'Pulp Fiction',
        'The Lord of the Rings',
        'Inception',
        'Fight Club'
    ],
    'genres': [
        'Action Sci-Fi',
        'Crime Drama',
        'Action Crime Drama',
        'Crime Drama',
        'Adventure Fantasy',
        'Action Sci-Fi Thriller',
        'Drama Thriller'
    ]
}

df = pd.DataFrame(data)


vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['genres'])


cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)



def recommend(title, cosine_sim=cosine_sim):
    if title not in df['title'].values:
        return ["Movie not found in dataset."]

    idx = df.index[df['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # земаме топ 3 препораки
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

if __name__=='__main__':

 print("Recommended movies for 'The Matrix':")
 print(recommend("The Matrix"))

 print("\nRecommended movies for 'The Godfather':")
 print(recommend("The Godfather"))
