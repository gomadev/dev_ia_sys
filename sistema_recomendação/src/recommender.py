import pandas as pd
from typing import List, Tuple, NamedTuple
from src.data_loader import DataLoader
from src.best_sellers import BestSellersRecommender, Recommendation as BestSellerRec
from src.similarity import CosineSimilarityRecommender, Recommendation as SimilarityRec


class Recommendation(NamedTuple):
    movie_id: int
    title: str
    score: float
    strategy: str
    explanation: str


class HybridRecommender:
    def __init__(self, data_dir: str = "data", min_ratings_threshold: int = 5, 
                 min_rating_for_similarity: float = 4.0):
        self.data_loader = DataLoader(data_dir)
        self.min_ratings_threshold = min_ratings_threshold
        self.min_rating_for_similarity = min_rating_for_similarity
        
        self.movies_df, self.ratings_df = self.data_loader.load_data()
        
        self.best_sellers = BestSellersRecommender(self.movies_df, self.ratings_df)
        self.similarity_recommender = CosineSimilarityRecommender(self.movies_df, self.ratings_df)
    
    def recommend(self, user_id: int, n: int = 10) -> List[Recommendation]:
        user_ratings = self.data_loader.get_user_ratings(user_id)
        
        if len(user_ratings) < self.min_ratings_threshold:
            return self._recommend_best_sellers(user_ratings, n)
        
        return self._recommend_personalized(user_ratings, n)
    
    def _recommend_best_sellers(self, user_ratings: pd.Series, n: int) -> List[Recommendation]:
        watched_movies = user_ratings.index.tolist()
        recommendations = self.best_sellers.recommend(n=n, exclude_movie_ids=watched_movies)
        
        return [
            Recommendation(
                movie_id=rec.movie_id,
                title=rec.title,
                score=rec.score,
                strategy="popular",
                explanation=rec.explanation
            )
            for rec in recommendations
        ]
    
    def _recommend_personalized(self, user_ratings: pd.Series, n: int) -> List[Recommendation]:
        recommendations = self.similarity_recommender.recommend_for_user(
            user_ratings, 
            n=n, 
            min_rating=self.min_rating_for_similarity
        )
        
        return [
            Recommendation(
                movie_id=rec.movie_id,
                title=rec.title,
                score=rec.score,
                strategy="similarity",
                explanation=rec.explanation
            )
            for rec in recommendations
        ]
    
    def get_user_stats(self, user_id: int) -> dict:
        user_ratings = self.data_loader.get_user_ratings(user_id)
        
        if len(user_ratings) == 0:
            return {
                "user_id": user_id,
                "total_ratings": 0,
                "avg_rating": 0.0,
                "recommendation_strategy": "none"
            }
        
        strategy = "popular" if len(user_ratings) < self.min_ratings_threshold else "similarity"
        
        return {
            "user_id": user_id,
            "total_ratings": len(user_ratings),
            "avg_rating": user_ratings.mean(),
            "recommendation_strategy": strategy,
            "top_rated_movies": self._get_top_rated_movies(user_ratings, n=5)
        }
    
    def _get_top_rated_movies(self, user_ratings: pd.Series, n: int = 5) -> List[Tuple[int, str, float]]:
        top_ratings = user_ratings.sort_values(ascending=False).head(n)
        
        return [
            (movie_id, self.data_loader.get_movie_title(movie_id), rating)
            for movie_id, rating in top_ratings.items()
        ]
