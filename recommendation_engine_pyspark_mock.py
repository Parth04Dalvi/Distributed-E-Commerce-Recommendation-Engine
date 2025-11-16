# --- Distributed E-Commerce Recommendation Engine (PySpark Mock) ---
# This script simulates a scalable recommendation system using Python's scientific stack
# (numpy/pandas) to represent the logic that would be executed efficiently on a PySpark cluster.
# This showcases skills in Big Data processing, Feature Engineering, and Machine Learning algorithms.

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# --- 1. MOCK DATA GENERATION (Simulates Distributed DataFrames) ---

def generate_mock_data(num_users=100, num_items=50, sparsity=0.8):
    """
    Generates a sparse User-Item interaction matrix (simulating data loaded via PySpark).
    
    Args:
        num_users (int): The number of users (rows).
        num_items (int): The number of items (columns).
        sparsity (float): The proportion of zero values (unrated items).
        
    Returns:
        pd.DataFrame: A User-Item interaction matrix.
    """
    print("Generating mock interaction data...")
    # Generate random ratings (1 to 5)
    ratings = np.random.randint(1, 6, size=(num_users, num_items))
    
    # Introduce sparsity (0 represents 'no interaction' or 'missing rating')
    mask = np.random.choice([0, 1], size=ratings.shape, p=[sparsity, 1 - sparsity])
    ratings[mask == 1] = 0 

    users = [f'user_{i+1}' for i in range(num_users)]
    items = [f'item_{i+1}' for i in range(num_items)]
    
    # DataFrame structure mimics the required format for matrix operations
    df = pd.DataFrame(ratings, index=users, columns=items)
    
    return df

# --- 2. CORE LOGIC: USER-BASED COLLABORATIVE FILTERING ---

def calculate_similarity_matrix(interaction_matrix):
    """
    Calculates the User-User Cosine Similarity matrix.
    This step simulates distributed matrix operations in PySpark (e.g., using BlockMatrix).
    
    Args:
        interaction_matrix (pd.DataFrame): The sparse User-Item interaction matrix.
        
    Returns:
        tuple: (User similarity DataFrame, Active interaction matrix)
    """
    print("Calculating User-User Cosine Similarity...")
    
    # Filter matrix to only include users with at least one rating (nonzero row sum)
    active_interaction_matrix = interaction_matrix.loc[(interaction_matrix!=0).any(axis=1)]
    
    # Calculate cosine similarity between user vectors (rows)
    user_similarity = cosine_similarity(active_interaction_matrix)
    
    # Convert similarity matrix back to DataFrame for indexing by user_id
    user_similarity_df = pd.DataFrame(
        user_similarity, 
        index=active_interaction_matrix.index, 
        columns=active_interaction_matrix.index
    )
    
    # Set similarity of a user to themselves to 0 to prevent self-recommendation bias
    np.fill_diagonal(user_similarity_df.values, 0)
    
    return user_similarity_df, active_interaction_matrix

def generate_recommendations(user_id, interaction_matrix, similarity_matrix, k_neighbors=5, n_recommendations=5):
    """
    Generates top N item recommendations for a target user using K-Nearest Neighbors (KNN).
    
    Args:
        user_id (str): The ID of the user to generate recommendations for.
        interaction_matrix (pd.DataFrame): The filtered User-Item matrix.
        similarity_matrix (pd.DataFrame): The User-User similarity matrix.
        k_neighbors (int): Number of nearest neighbors to consider.
        n_recommendations (int): Number of top items to recommend.
        
    Returns:
        list: A list of (item_id, predicted_rating) tuples.
    """
    print(f"\nGenerating recommendations for {user_id} (K={k_neighbors})...")
    
    if user_id not in similarity_matrix.index:
        return f"Error: User {user_id} not found in active dataset."

    # 1. Find K-Nearest Neighbors (KNN)
    user_similarities = similarity_matrix[user_id].sort_values(ascending=False)
    knn = user_similarities.head(k_neighbors).index.tolist()
    
    print(f"Nearest Neighbors found: {knn}")

    # 2. Identify unrated items (candidates for recommendation)
    rated_items = interaction_matrix.loc[user_id][interaction_matrix.loc[user_id] != 0].index.tolist()
    unrated_items = interaction_matrix.columns.difference(rated_items)
    
    # 3. Calculate predicted ratings for unrated items (Weighted Sum)
    predictions = defaultdict(float)
    similarity_sum = defaultdict(float)
    
    for item in unrated_items:
        for neighbor in knn:
            neighbor_rating = interaction_matrix.loc[neighbor, item]
            
            # Only consider neighbors who have rated this specific item
            if neighbor_rating > 0:
                sim_score = similarity_matrix.loc[user_id, neighbor]
                
                # Weighted Sum component: (Similarity * Neighbor_Rating)
                predictions[item] += sim_score * neighbor_rating
                # Sum of weights component: (Similarity)
                similarity_sum[item] += sim_score
    
    # Finalize predictions by dividing the weighted sum by the sum of similarities
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

    # 1. Load Data (Mock)
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
