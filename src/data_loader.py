import pandas as pd
from pathlib import Path
from typing import Tuple


class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.movies_df = None
        self.ratings_df = None
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.movies_df = pd.read_csv(self.data_dir / "movies.csv")
        self.ratings_df = pd.read_csv(self.data_dir / "ratings.csv")
        return self.movies_df, self.ratings_df
    
    def get_user_movie_matrix(self) -> pd.DataFrame:
        if self.ratings_df is None:
            self.load_data()
        
        return self.ratings_df.pivot_table(
            index='userId',
            columns='movieId',
            values='rating'
        )
    
    def get_movie_user_matrix(self) -> pd.DataFrame:
        if self.ratings_df is None:
            self.load_data()
        
        return self.ratings_df.pivot_table(
            index='movieId',
            columns='userId',
            values='rating'
        )
    
    def get_user_ratings(self, user_id: int) -> pd.Series:
        if self.ratings_df is None:
            self.load_data()
        
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        return user_ratings.set_index('movieId')['rating']
    
    def get_movie_title(self, movie_id: int) -> str:
        if self.movies_df is None:
            self.load_data()
        
        movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
        return movie_row['title'].values[0] if not movie_row.empty else f"Movie {movie_id}"
