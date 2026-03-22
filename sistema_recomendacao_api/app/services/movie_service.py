import pandas as pd
import os
from app.models.movie import Movie
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MOVIES_PATH = os.path.join(DATA_DIR, 'movies.csv')

class MovieService:
    @staticmethod
    def get_all_movies() -> List[Movie]:
        df = pd.read_csv(MOVIES_PATH)
        return [Movie(**row) for row in df.to_dict(orient="records")]

    @staticmethod
    def get_movie_by_id(movie_id: int) -> Movie | None:
        df = pd.read_csv(MOVIES_PATH)
        movie = df[df['movieId'] == movie_id]
        if movie.empty:
            return None
        return Movie(**movie.iloc[0].to_dict())

    @staticmethod
    def create_movie(movie: Movie) -> Movie:
        df = pd.read_csv(MOVIES_PATH)
        if movie.movieId in df['movieId'].values:
            raise ValueError("Filme já existe")
        df = pd.concat([df, pd.DataFrame([movie.dict()])], ignore_index=True)
        df.to_csv(MOVIES_PATH, index=False)
        return movie

    @staticmethod
    def update_movie(movie_id: int, movie: Movie) -> Movie | None:
        df = pd.read_csv(MOVIES_PATH)
        idx = df.index[df['movieId'] == movie_id].tolist()
        if not idx:
            return None
        df.loc[idx[0], ['title', 'genres']] = movie.title, movie.genres
        df.to_csv(MOVIES_PATH, index=False)
        return Movie(**df.loc[idx[0]].to_dict())

    @staticmethod
    def delete_movie(movie_id: int) -> bool:
        df = pd.read_csv(MOVIES_PATH)
        idx = df.index[df['movieId'] == movie_id].tolist()
        if not idx:
            return False
        df = df.drop(idx[0])
        df.to_csv(MOVIES_PATH, index=False)
        return True
