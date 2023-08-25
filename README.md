# Content-Based Article Recommender System
This project demonstrates the implementation of a content-based recommender system for articles. The recommendation is based on the content in the Description column of the articles.csv dataset. It further incorporates enhancements to boost the recommendation score of articles by the same author and for more recent articles.
Developed a content-based recommendation system using TF-IDF vectorization and cosine similarity to suggest articles based on textual descriptions.

## Dependencies
pandas: For data manipulation.
sklearn: Specifically, TfidfVectorizer for text vectorization and linear_kernel for similarity calculation.
To install these dependencies, you can use pip:

pip install pandas scikit-learn

## Dataset Overview
The dataset (articles.csv) contains the following columns:

ArticleID: Unique identifier for each article.
PublishedDate: The date the article was published.
MainAuthor: Main author of the article.
Description: Brief description or abstract of the article.

## Implementation Details
### Data Preprocessing
1) Missing values in the Description column are replaced with an empty string.
2) PublishedDate column is converted to a DateTime object.
   
### TF-IDF Vectorization
The Description column is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency) to transform the text into a numerical format suitable for similarity calculations.

### Cosine Similarity Calculation
Using the TF-IDF matrix, cosine similarity scores between articles are computed. This score is the basis for our content-based recommendation.

### Recommendation Function
The recommend function returns articles similar to a given article ID. Recommendations can be enhanced in two ways:

1) Articles by the same author get a boosted score.
2) More recent articles also get a boosted score.

## Usage
To get recommendations for a specific article, simply call the recommend function with the desired ArticleID and the number of recommendations:

recommendations = recommend(ArticleID, num_recommendations)
print_recommendations(ArticleID, recommendations)
