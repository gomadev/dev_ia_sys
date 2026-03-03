import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from src.recommender import HybridRecommender


class RecommenderEvaluator:
    def __init__(self, recommender: HybridRecommender, test_size: float = 0.2, 
                 random_seed: int = 42):
        self.recommender = recommender
        self.test_size = test_size
        self.random_seed = random_seed
        self.train_ratings = None
        self.test_ratings = None
    
    def train_test_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        np.random.seed(self.random_seed)
        
        ratings_df = self.recommender.ratings_df.copy()
        
        user_test_data = []
        user_train_data = []
        
        for user_id in ratings_df['userId'].unique():
            user_ratings = ratings_df[ratings_df['userId'] == user_id]
            
            if len(user_ratings) < 5:
                user_train_data.append(user_ratings)
                continue
            
            n_test = max(1, int(len(user_ratings) * self.test_size))
            test_indices = np.random.choice(user_ratings.index, size=n_test, replace=False)
            
            test_data = user_ratings.loc[test_indices]
            train_data = user_ratings.drop(test_indices)
            
            user_test_data.append(test_data)
            user_train_data.append(train_data)
        
        self.train_ratings = pd.concat(user_train_data, ignore_index=True)
        self.test_ratings = pd.concat(user_test_data, ignore_index=True)
        
        return self.train_ratings, self.test_ratings
    
    def precision_at_k(self, user_id: int, k: int = 10, 
                       relevant_threshold: float = 4.0) -> float:
        if self.test_ratings is None:
            self.train_test_split()
        
        user_test = self.test_ratings[self.test_ratings['userId'] == user_id]
        relevant_items = set(
            user_test[user_test['rating'] >= relevant_threshold]['movieId'].tolist()
        )
        
        if len(relevant_items) == 0:
            return 0.0
        
        user_train = self.train_ratings[self.train_ratings['userId'] == user_id]
        user_train_series = user_train.set_index('movieId')['rating']
        
        try:
            recommendations = self.recommender.similarity_recommender.recommend_for_user(
                user_train_series, n=k
            )
            recommended_items = [rec.movie_id for rec in recommendations]
        except:
            return 0.0
        
        hits = len(set(recommended_items) & relevant_items)
        return hits / k if k > 0 else 0.0
    
    def recall_at_k(self, user_id: int, k: int = 10, 
                    relevant_threshold: float = 4.0) -> float:
        if self.test_ratings is None:
            self.train_test_split()
        
        user_test = self.test_ratings[self.test_ratings['userId'] == user_id]
        relevant_items = set(
            user_test[user_test['rating'] >= relevant_threshold]['movieId'].tolist()
        )
        
        if len(relevant_items) == 0:
            return 0.0
        
        user_train = self.train_ratings[self.train_ratings['userId'] == user_id]
        user_train_series = user_train.set_index('movieId')['rating']
        
        try:
            recommendations = self.recommender.similarity_recommender.recommend_for_user(
                user_train_series, n=k
            )
            recommended_items = [rec.movie_id for rec in recommendations]
        except:
            return 0.0
        
        hits = len(set(recommended_items) & relevant_items)
        return hits / len(relevant_items)
    
    def f1_score_at_k(self, user_id: int, k: int = 10, 
                      relevant_threshold: float = 4.0) -> float:
        precision = self.precision_at_k(user_id, k, relevant_threshold)
        recall = self.recall_at_k(user_id, k, relevant_threshold)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate_all_users(self, k: int = 10, sample_size: int = None) -> Dict[str, float]:
        if self.test_ratings is None:
            self.train_test_split()
        
        users = self.test_ratings['userId'].unique()
        
        if sample_size and len(users) > sample_size:
            users = np.random.choice(users, size=sample_size, replace=False)
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        for user_id in users:
            user_train = self.train_ratings[self.train_ratings['userId'] == user_id]
            
            if len(user_train) < 5:
                continue
            
            precision = self.precision_at_k(user_id, k)
            recall = self.recall_at_k(user_id, k)
            f1 = self.f1_score_at_k(user_id, k)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
        
        return {
            'precision@k': np.mean(precision_scores) if precision_scores else 0.0,
            'recall@k': np.mean(recall_scores) if recall_scores else 0.0,
            'f1@k': np.mean(f1_scores) if f1_scores else 0.0,
            'num_users_evaluated': len(precision_scores),
            'k': k
        }
    
    def coverage(self, k: int = 10, sample_users: int = 100) -> float:
        users = self.recommender.ratings_df['userId'].unique()
        
        if len(users) > sample_users:
            users = np.random.choice(users, size=sample_users, replace=False)
        
        recommended_movies = set()
        
        for user_id in users:
            try:
                recommendations = self.recommender.recommend(user_id, n=k)
                for rec in recommendations:
                    recommended_movies.add(rec.movie_id)
            except:
                continue
        
        total_movies = len(self.recommender.movies_df)
        return len(recommended_movies) / total_movies
    
    def diversity(self, k: int = 10, sample_users: int = 100) -> float:
        users = self.recommender.ratings_df['userId'].unique()
        
        if len(users) > sample_users:
            users = np.random.choice(users, size=sample_users, replace=False)
        
        all_recommendations = []
        
        for user_id in users:
            try:
                recommendations = self.recommender.recommend(user_id, n=k)
                movie_ids = [rec.movie_id for rec in recommendations]
                all_recommendations.extend(movie_ids)
            except:
                continue
        
        if len(all_recommendations) == 0:
            return 0.0
        
        unique_recommendations = len(set(all_recommendations))
        total_recommendations = len(all_recommendations)
        
        return unique_recommendations / total_recommendations
