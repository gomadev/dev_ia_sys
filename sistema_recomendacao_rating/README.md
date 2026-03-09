# Sistema de Recomendacao de Filmes — Baseado em Rating

Sistema hibrido que combina duas estrategias, usando a **nota (rating) dos usuarios** como dado principal:

- **Popularidade por Rating**: Filmes com melhor nota media, ponderando tambem a quantidade de avaliacoes
- **Similaridade de Cossenos ponderada por Rating**: Recomendacoes personalizadas onde a nota do usuario amplifica ou reduz a influencia de cada filme fonte

## Como o Rating e usado

| Componente | Uso do Rating |
|---|---|
| `best_sellers.py` | Score = `avg_rating * 0.7 + popularidade_normalizada * 0.3` |
| `similarity.py` | `score = similaridade_cosseno * (nota_do_usuario / 5.0)` |
| `metrics.py` | Relevancia definida como `rating >= 4.0` |

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
├── data_loader.py      # Carrega movies.csv e ratings.csv
├── best_sellers.py     # Recomendacoes por melhor nota media
├── similarity.py       # Similaridade de cossenos ponderada por rating
├── recommender.py      # Sistema hibrido
└── metrics.py          # Precision@K, Recall@K, F1@K, Cobertura, Diversidade
```
