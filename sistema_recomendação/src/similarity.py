import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, NamedTuple


class Recommendation(NamedTuple):
    movie_id: int
    title: str
    score: float
    explanation: str


class CosineSimilarityRecommender:
    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.movie_user_matrix = None
        self.similarity_matrix = None
    
    def build_similarity_matrix(self) -> np.ndarray:
        if self.similarity_matrix is not None:
            return self.similarity_matrix
        
        self.movie_user_matrix = self.ratings_df.pivot_table(
            index='movieId',
            columns='userId',
            values='rating',
            fill_value=0
        )
        
        self.similarity_matrix = cosine_similarity(self.movie_user_matrix)
        return self.similarity_matrix
    
    def get_similar_movies(self, movie_id: int, n: int = 10) -> List[Tuple[int, float]]:
        if self.similarity_matrix is None:
            self.build_similarity_matrix()
        
        movie_ids = self.movie_user_matrix.index.tolist()
        
        if movie_id not in movie_ids:
            return []
        
        movie_idx = movie_ids.index(movie_id)
        similarities = self.similarity_matrix[movie_idx]
        
        similar_indices = np.argsort(similarities)[::-1][1:n+1]
        
        return [
            (movie_ids[idx], similarities[idx])
            for idx in similar_indices
        ]
    
    def recommend_for_user(self, user_ratings: pd.Series, n: int = 10, 
                          min_rating: float = 4.0) -> List[Recommendation]:
        if self.similarity_matrix is None:
            self.build_similarity_matrix()
        
        liked_movies = user_ratings[user_ratings >= min_rating].index.tolist()
        watched_movies = set(user_ratings.index.tolist())
        
        candidate_scores: Dict[int, List[Tuple[int, float]]] = {}
        
        for movie_id in liked_movies:
            user_rating = float(user_ratings[movie_id])
            rating_weight = user_rating / 5.0  # normaliza nota do usuario para [0, 1]
            similar_movies = self.get_similar_movies(movie_id, n=50)
            
            for similar_movie_id, similarity_score in similar_movies:
                if similar_movie_id not in watched_movies:
                    if similar_movie_id not in candidate_scores:
                        candidate_scores[similar_movie_id] = []
                    # pondera a similaridade pela nota que o usuario deu ao filme fonte
                    candidate_scores[similar_movie_id].append((movie_id, similarity_score * rating_weight))
        
        recommendations = self._build_recommendations_with_explanations(
            candidate_scores, user_ratings, n
        )
        return recommendations
    
    def _build_recommendations_with_explanations(
        self, 
        scores: Dict[int, List[Tuple[int, float]]], 
        user_ratings: pd.Series,
        n: int
    ) -> List[Recommendation]:
        aggregated = []
        
        for movie_id, similar_to_list in scores.items():
            avg_score = np.mean([score for _, score in similar_to_list])
            
            top_contributors = sorted(similar_to_list, key=lambda x: x[1], reverse=True)[:3]
            
            explanation = self._create_explanation(top_contributors, user_ratings)
            
            aggregated.append((movie_id, avg_score, explanation))
        
        aggregated.sort(key=lambda x: x[1], reverse=True)
        top_n = aggregated[:n]
        
        return [
            Recommendation(
                movie_id=movie_id,
                title=self._get_movie_title(movie_id),
                score=score,
                explanation=explanation
            )
            for movie_id, score, explanation in top_n
        ]
    
    def _create_explanation(
        self, 
        contributors: List[Tuple[int, float]], 
        user_ratings: pd.Series
    ) -> str:
        movie_names = []
        
        for movie_id, weighted_score in contributors[:2]:
            title = self._get_movie_title(movie_id)
            rating = user_ratings.get(movie_id, 0)
            movie_names.append(f"{title} (nota: {rating:.1f}, score: {weighted_score:.3f})")
        
        if len(movie_names) == 1:
            return f"Similar a: {movie_names[0]}"
        else:
            return f"Similar a: {', '.join(movie_names)}"
    
    def _aggregate_scores(self, scores: Dict[int, List[float]], n: int) -> List[Tuple[int, str, float]]:
        aggregated = [
            (movie_id, np.mean(similarities))
            for movie_id, similarities in scores.items()
        ]
        
        aggregated.sort(key=lambda x: x[1], reverse=True)
        
        top_n = aggregated[:n]
        
        return [
            (movie_id, self._get_movie_title(movie_id), score)
            for movie_id, score in top_n
        ]
    
    def _get_movie_title(self, movie_id: int) -> str:
        movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
        return movie_row['title'].values[0] if not movie_row.empty else f"Movie {movie_id}"
