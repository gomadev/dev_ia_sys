# Sistema de Recomendação de Filmes - API

API desenvolvida com FastAPI para servir dados de filmes a partir de um arquivo CSV.

## Estrutura Modular
- **app/main.py**: Inicialização da aplicação FastAPI.
- **app/routers/movies.py**: Rotas relacionadas a filmes.
- **app/services/movie_service.py**: Lógica de acesso e manipulação dos dados.
- **app/models/movie.py**: Modelos Pydantic para validação e documentação.

## Como rodar

1. Crie o ambiente virtual:
   ```bash
   python -m venv .venv
   ```
2. Ative o ambiente virtual:
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```
3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
4. Rode a API:
   ```bash
   uvicorn app.main:app --reload
   ```

http://localhost:8000/docs
