import pandas as pd
from typing import List, NamedTuple


class Recommendation(NamedTuple):
    movie_id: int
    title: str
    score: float
    explanation: str


class BestSellersRecommender:
    """
    Recomenda filmes mais bem avaliados usando a nota media (rating) como
    criterio principal, com a quantidade de avaliacoes como fator secundario.

    Formula: score = avg_rating * 0.7 + (num_ratings / max_ratings) * 5 * 0.3
    """

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame,
                 min_ratings: int = 50):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        self.min_ratings = min_ratings
        self.best_sellers_cache = None

    def calculate_best_sellers(self) -> pd.DataFrame:
        if self.best_sellers_cache is not None:
            return self.best_sellers_cache

        # Agrupa por filme calculando media e contagem de ratings
        movie_stats = self.ratings_df.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            num_ratings=('rating', 'count')
        ).reset_index()

        # Filtra apenas filmes com avaliacoes suficientes
        popular_movies = movie_stats[movie_stats['num_ratings'] >= self.min_ratings].copy()

        # Score: 70% da nota media + 30% da popularidade normalizada
        # A nota media (rating) e o dado principal conforme requisito
        popular_movies['score'] = (
            popular_movies['avg_rating'] * 0.7 +
            (popular_movies['num_ratings'] / popular_movies['num_ratings'].max()) * 5 * 0.3
        )

        best_sellers = popular_movies.sort_values('score', ascending=False)
        best_sellers = best_sellers.merge(self.movies_df, on='movieId', how='left')

        self.best_sellers_cache = best_sellers
        return best_sellers

    def recommend(self, n: int = 10, exclude_movie_ids: List[int] = None) -> List[Recommendation]:
        best_sellers = self.calculate_best_sellers()

        if exclude_movie_ids:
            best_sellers = best_sellers[~best_sellers['movieId'].isin(exclude_movie_ids)]

        top_n = best_sellers.head(n)

        return [
            Recommendation(
                movie_id=row['movieId'],
                title=row['title'],
                score=row['score'],
                explanation=(
                    f"Nota media: {row['avg_rating']:.2f}/5.0 | "
                    f"Avaliacoes: {int(row['num_ratings'])}"
                )
            )
            for _, row in top_n.iterrows()
        ]
