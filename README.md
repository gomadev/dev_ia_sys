# Sistema de Recomendacao de Filmes

Sistema hibrido que combina duas estrategias:
- **Popularidade**: Filmes mais assistidos e bem avaliados
- **Similaridade de Cossenos**: Recomendacoes personalizadas baseadas no perfil do usuario


## Instalacao

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Testar recomendacoes
python main.py

# Avaliar metricas do modelo
python evaluate.py
```

## Estrutura

```
src/
├── data_loader.py      # Carrega dados
├── best_sellers.py     # Recomendacoes populares
├── similarity.py       # Similaridade de cossenos
├── recommender.py      # Sistema hibrido
└── metrics.py          # Metricas de avaliacao
```
