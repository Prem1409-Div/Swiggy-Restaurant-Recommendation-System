import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load preprocessed data and transformer
df = pd.read_csv('preprocessed_swiggy_data.csv')
preprocessor = joblib.load('preprocessor.joblib')

# Get feature matrix for similarity comparison
def get_feature_matrix(df):
    categorical_cols = ['city', 'cuisine']
    numerical_cols = ['rating', 'rating_count', 'cost']
    X = preprocessor.transform(df[categorical_cols + numerical_cols])
    return X

# Recommend top N restaurants
def recommend(city, cuisine, rating, cost, top_n=5):
    user_input = pd.DataFrame([{
        'city': city,
        'cuisine': cuisine,
        'rating': rating,
        'rating_count': 0,  # neutral value
        'cost': cost
    }])

    # Use same preprocessing
    user_vector = preprocessor.transform(user_input)

    # Restaurant vectors
    restaurant_vectors = get_feature_matrix(df)

    # Compute cosine similarity
    similarity_scores = cosine_similarity(user_vector, restaurant_vectors).flatten()

    # Get top N indices
    top_indices = similarity_scores.argsort()[::-1][:top_n]

    # Return top N recommendations
    return df.iloc[top_indices][['name', 'city', 'cuisine', 'rating', 'cost', 'address']]

# Test
if __name__ == "__main__":
    city_input = "Chennai"
    cuisine_input = "South Indian"
    rating_input = 4.0
    cost_input = 300

    recommendations = recommend(city_input, cuisine_input, rating_input, cost_input)
    print("🔍 Top Recommendations:")
    print(recommendations.to_string(index=False))
