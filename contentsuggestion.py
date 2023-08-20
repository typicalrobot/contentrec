#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


import pandas as pd


articles_df = pd.read_csv('articles.csv')


articles_df['PublishedDate'] = pd.to_datetime(articles_df['PublishedDate'])


print(articles_df)


# In[5]:


# Replace NaN values in 'Description' column with an empty string
articles_df['Description'] = articles_df['Description'].fillna('')

# Continue with your TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(articles_df['Description'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(articles_df['Description'])

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[7]:


def recommend(article_id, num_recommendations, cosine_similarities=cosine_similarities):
    article_idx = articles_df.index[articles_df['ArticleID'] == article_id].tolist()[0]
    
    # Get this article's data
    article_data = articles_df.loc[article_idx]
    
    # Get the initial content-based similarities
    similar_articles = list(enumerate(cosine_similarities[article_idx]))
    
    for i, (_, similarity) in enumerate(similar_articles):
        # Boost similarity score for articles by the same author
        if articles_df.loc[i, 'MainAuthor'] == article_data['MainAuthor']:
            similarity += 0.1
            
        # Boost similarity score for more recent articles
        if articles_df.loc[i, 'PublishedDate'] > article_data['PublishedDate']:
            similarity += 0.05
            
        similar_articles[i] = (i, similarity)
    
    # Sort the articles based on the adjusted similarity scores
    similar_articles = sorted(similar_articles, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the `num_recommendations` most similar articles
    recommended_articles = [articles_df.iloc[index]['ArticleID'] for index, similarity in similar_articles[1:num_recommendations+1]]
    
    return recommended_articles

# Example: Recommend 3 articles similar to the article with ID 1
recommended_for_physics = recommend(1, 3)
print(recommended_for_physics)


# In[8]:


def print_recommendations(input_id, recommendations):
    print(f"Recommendations for '{articles_df.loc[articles_df['ArticleID'] == input_id, 'ArticleTitle'].iloc[0]}':\n")
    for rec_id in recommendations:
        print(f"- {articles_df.loc[articles_df['ArticleID'] == rec_id, 'ArticleTitle'].iloc[0]}")

print_recommendations(1, recommended_for_physics)

