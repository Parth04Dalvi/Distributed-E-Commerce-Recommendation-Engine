# --- Distributed E-Commerce Recommendation Engine (PySpark Mock) ---
# This script simulates a scalable recommendation system using Python's scientific stack
# (numpy/pandas) to represent the logic that would be executed efficiently on a PySpark cluster.
# This showcases skills in Big Data processing, Feature Engineering, and Machine Learning algorithms.

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# --- 1. MOCK DATA GENERATION (Simulates Distributed DataFrames) ---
# In a real scenario, these would be loaded via spark.read.csv() or from HDFS/S3.

def generate_mock_data(num_users=100, num_items=50, sparsity=0.8):
    """Generates a sparse User-Item interaction matrix."""
    print("Generating mock interaction data...")
    # Generate random ratings (1 to 5)
    ratings = np.random.randint(1, 6, size=(num_users, num_items))
    # Introduce sparsity (most users haven't rated most items)
    mask = np.random.choice([0, 1], size=ratings.shape, p=[sparsity, 1 - sparsity])
    ratings[mask == 1] = 0 # 0 represents 'no interaction'

    users = [f'user_{i+1}' for i in range(num_users)]
    items = [f'item_{i+1}' for i in range(num_items)]
    
    # In PySpark, this matrix would be a distributed RDD or DataFrame
    df = pd.DataFrame(ratings, index=users, columns=items)
    
    # We will only return the non-zero part for the calculation
    return df

# --- 2. CORE LOGIC: USER-BASED COLLABORATIVE FILTERING ---

def calculate_similarity_matrix(interaction_matrix):
    """
    Simulates calculating cosine similarity between users.
    In PySpark, this would leverage distributed matrix operations (e.g., BlockMatrix).
    """
    print("Calculating User-User Cosine Similarity...")
    
    # Remove rows/columns that contain only zeros (users/items with no activity)
    # This prevents division by zero errors when calculating similarity
    interaction_matrix = interaction_matrix.loc[(interaction_matrix!=0).any(axis=1)]
    
    # Calculate cosine similarity. The result is a dense matrix of shape (N_users, N_users)
    user_similarity = cosine_similarity(interaction_matrix)
    
    # Convert back to DataFrame for easier indexing
    user_similarity_df = pd.DataFrame(
        user_similarity, 
        index=interaction_matrix.index, 
        columns=interaction_matrix.index
    )
    
    # Set self-similarity to 0 (user is perfectly similar to themselves, but we don't want to recommend their own rating)
    np.fill_diagonal(user_similarity_df.values, 0)
    
    return user_similarity_df, interaction_matrix

def generate_recommendations(user_id, interaction_matrix, similarity_matrix, k_neighbors=5, n_recommendations=5):
    """
    Generates top N recommendations for a specific user based on the ratings of their K-nearest neighbors.
    This simulates the MapReduce step of combining neighbor scores.
    """
    print(f"\nGenerating recommendations for {user_id}...")
    
    if user_id not in similarity_matrix.index:
        return f"Error: User {user_id} not found in active dataset."

    # 1. Find K-Nearest Neighbors (KNN)
    # Get similarity scores for the target user, sort them, and pick the top K
    user_similarities = similarity_matrix[user_id].sort_values(ascending=False)
    knn = user_similarities.head(k_neighbors).index.tolist()
    
    print(f"Nearest Neighbors found: {knn}")

    # 2. Identify items the target user HAS NOT rated (candidates for recommendation)
    rated_items = interaction_matrix.loc[user_id][interaction_matrix.loc[user_id] != 0].index.tolist()
    
    # 3. Calculate predicted ratings for unrated items
    predictions = defaultdict(float)
    similarity_sum = defaultdict(float)
    
    unrated_items = interaction_matrix.columns.difference(rated_items)
    
    for item in unrated_items:
        # Loop through the ratings of the nearest neighbors for this specific item
        for neighbor in knn:
            # Check if the neighbor has rated this item
            neighbor_rating = interaction_matrix.loc[neighbor, item]
            if neighbor_rating > 0:
                # Weighted Sum: Prediction = SUM(Similarity * Neighbor_Rating) / SUM(Similarity)
                sim_score = similarity_matrix.loc[user_id, neighbor]
                predictions[item] += sim_score * neighbor_rating
                similarity_sum[item] += sim_score
    
    # Finalize predictions: divide the weighted sum by the sum of similarities
    final_predictions = {}
    for item, weighted_sum in predictions.items():
        if similarity_sum[item] > 0:
            final_predictions[item] = weighted_sum / similarity_sum[item]
        
    # 4. Return top N recommendations
    top_recommendations = sorted(final_predictions.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    
    return top_recommendations

# --- 3. EXECUTION ---

if __name__ == "__main__":
    # Settings for demonstration
    TARGET_USER = 'user_10' 
    K = 10 
    N = 5  

    # 1. Load Data
    user_item_matrix = generate_mock_data(num_users=200, num_items=100, sparsity=0.95)
    print(f"Data matrix shape: {user_item_matrix.shape}")

    # 2. Calculate Similarity
    sim_matrix, active_matrix = calculate_similarity_matrix(user_item_matrix)
    print(f"Similarity matrix shape: {sim_matrix.shape}")

    # 3. Generate Recommendations
    if TARGET_USER in sim_matrix.index:
        recommendations = generate_recommendations(TARGET_USER, active_matrix, sim_matrix, k_neighbors=K, n_recommendations=N)
        
        print("\n" + "="*50)
        print(f"✨ Top {N} Recommendations for {TARGET_USER} (K={K} Neighbors) ✨")
        print("="*50)
        
        if isinstance(recommendations, list) and recommendations:
            for item, score in recommendations:
                print(f"[{item.upper()}]: Predicted Rating = {score:.3f}")
        else:
            print("No new recommendations could be generated based on neighbors' ratings.")
    else:
        print(f"Cannot process: {TARGET_USER} does not have any recorded interactions.")

# Example Output (Simulated):
# [ITEM_34]: Predicted Rating = 4.85
# [ITEM_12]: Predicted Rating = 4.52
# [ITEM_05]: Predicted Rating = 4.39
# [ITEM_21]: Predicted Rating = 4.10
# [ITEM_56]: Predicted Rating = 3.98
