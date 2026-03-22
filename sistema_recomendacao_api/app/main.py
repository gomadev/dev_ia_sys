from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import movies
from app.logging_middleware import LoggingMiddleware

app = FastAPI(title="Sistema de Recomendação de Filmes", description="API para acessar dados de filmes.", version="1.0.0")
app.add_middleware(LoggingMiddleware)

# Configuração de CORS seguro
origins = [
    "https://sara-git-desenvolvimento-guilhermegm-5117s-projects.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclui o router de filmes
app.include_router(movies.router)

# Rota raiz para mensagem amigável
@app.get("/", tags=["Root"])
def root():
    return {"message": "API. /docs para a doc."}
