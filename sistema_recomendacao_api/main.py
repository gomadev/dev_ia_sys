from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import os

app = FastAPI(title="Sistema de Recomendação de Filmes", description="API para acessar dados de filmes.", version="1.0.0")

# Caminho para os dados
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MOVIES_PATH = os.path.join(DATA_DIR, 'movies.csv')

# Carregar dados dos filmes
movies_df = pd.read_csv(MOVIES_PATH)

@app.get("/movies", summary="Listar todos os filmes")
def get_movies():
    """Retorna todos os filmes disponíveis."""
    return JSONResponse(content=movies_df.to_dict(orient="records"))

@app.get("/movies/{movie_id}", summary="Obter detalhes de um filme")
def get_movie(movie_id: int):
    """Retorna detalhes de um filme pelo seu ID."""
    movie = movies_df[movies_df['movieId'] == movie_id]
    if movie.empty:
        return JSONResponse(status_code=404, content={"error": "Filme não encontrado."})
    return JSONResponse(content=movie.iloc[0].to_dict())
