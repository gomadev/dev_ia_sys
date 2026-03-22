from fastapi import APIRouter, HTTPException, Depends, status
from app.services.movie_service import MovieService
from app.models.movie import Movie
from typing import List

router = APIRouter(prefix="/movies", tags=["Movies"])

from fastapi import Query

@router.get(
    "/",
    response_model=List[Movie],
    summary="Listar todos os filmes",
    response_description="Lista de filmes",
    responses={
        200: {"description": "Lista de filmes", "content": {"application/json": {"example": [{"movieId": 1, "title": "Toy Story (1995)", "genres": "Adventure|Animation|Children|Comedy|Fantasy"}]}}}
    }
)
def list_movies(
    skip: int = Query(0, ge=0, description="Índice inicial para paginação"),
    limit: int = Query(10, ge=1, le=100, description="Quantidade máxima de filmes"),
    title: str = Query(None, description="Filtrar por título"),
    genre: str = Query(None, description="Filtrar por gênero")
):
    """Lista filmes com paginação e filtros."""
    movies = MovieService.get_all_movies()
    if title:
        movies = [m for m in movies if title.lower() in m.title.lower()]
    if genre:
        movies = [m for m in movies if genre.lower() in m.genres.lower()]
    return movies[skip:skip+limit]

@router.get("/{movie_id}", response_model=Movie, summary="Obter detalhes de um filme")
def get_movie(movie_id: int):
    movie = MovieService.get_movie_by_id(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail="Filme não encontrado.")
    return movie

# CRUD protegido
@router.post(
    "/",
    response_model=Movie,
    status_code=status.HTTP_201_CREATED,
    summary="Criar novo filme",
    response_description="Filme criado com sucesso",
    responses={
        201: {"description": "Filme criado", "content": {"application/json": {"example": {"movieId": 999, "title": "Novo Filme", "genres": "Ação|Aventura"}}}},
        400: {"description": "Filme já existe"}
    }
)
def create_movie(movie: Movie):
    """Cria um novo filme."""
    try:
        return MovieService.create_movie(movie)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/{movie_id}", response_model=Movie, summary="Atualizar filme")
def update_movie(movie_id: int, movie: Movie):
    updated = MovieService.update_movie(movie_id, movie)
    if not updated:
        raise HTTPException(status_code=404, detail="Filme não encontrado.")
    return updated

@router.delete("/{movie_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Deletar filme")
def delete_movie(movie_id: int):
    deleted = MovieService.delete_movie(movie_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Filme não encontrado.")
