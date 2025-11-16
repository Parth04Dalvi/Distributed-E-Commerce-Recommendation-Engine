# Distributed-E-Commerce-Recommendation-Engine

<img width="724" height="646" alt="image" src="https://github.com/user-attachments/assets/84337de0-0624-412c-aa07-53963fc2b53a" />

Distributed E-Commerce Recommendation Engine (PySpark Mock) 🛒

Overview

This project implements the core logic for a User-Based Collaborative Filtering recommendation system. It is focused on demonstrating Big Data & ML principles for scalable execution in a PySpark environment.

It showcases proficiency in handling sparse data, distributed matrix operations, and applying classical Machine Learning techniques to generate personalized product recommendations.

The script uses Python (Pandas/NumPy) to mock the data structures and algorithms that would be executed across a cluster, highlighting the design pattern for a production-ready solution.

Key Features

Scalable Architecture Mock: Uses Python (Pandas/NumPy) to mimic the distributed handling of a large, sparse User-Item interaction matrix.

User-Item Matrix Generation: Simulates the creation of a realistic large-scale interaction matrix typical of e-commerce rating data.

Collaborative Filtering: Implements the core logic to identify K-Nearest Neighbors (KNN) based on user behavior.

Cosine Similarity Calculation: Efficiently computes the similarity between all active users, simulating distributed matrix algebra.

Recommendation Generation: Calculates predicted item ratings for unrated items using a weighted sum approach.

Execution

This script is runnable in any standard Python environment with pandas and numpy installed (which are required by the sklearn.metrics.pairwise.cosine_similarity module).

python recommendation_engine_pyspark_mock.py

The output demonstrates the end-to-end process: generating mock data, calculating similarity matrices, and listing the top predicted items for a target user.
