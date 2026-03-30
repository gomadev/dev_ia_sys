from src.recommender import HybridRecommender
from src.metrics import RecommenderEvaluator


def print_separator():
    print("\n" + "=" * 80 + "\n")


def main():
    print("Inicializando sistema de recomendacao para avaliacao...")
    recommender = HybridRecommender(data_dir="data")
    
    print("Construindo matriz de similaridade...")
    recommender.similarity_recommender.build_similarity_matrix()
    print("Matriz construida com sucesso!")
    
    print_separator()
    print("Criando avaliador e dividindo dados em treino/teste (80/20)...")
    evaluator = RecommenderEvaluator(recommender, test_size=0.2, random_seed=42)
    train, test = evaluator.train_test_split()
    
    print(f"Dados de treino: {len(train)} avaliacoes")
    print(f"Dados de teste: {len(test)} avaliacoes")
    
    print_separator()
    print("Avaliando modelo com metricas Precision@K, Recall@K e F1@K...")
    print("(Avaliando amostra de usuarios para melhor performance)")
    
    k_values = [5, 10, 20]
    
    for k in k_values:
        print(f"\nAvaliando com K={k}...")
        metrics = evaluator.evaluate_all_users(k=k, sample_size=50)
        
        print(f"\nResultados para K={k}:")
        print(f"  Precision@{k}: {metrics['precision@k']:.4f}")
        print(f"  Recall@{k}:    {metrics['recall@k']:.4f}")
        print(f"  F1-Score@{k}:  {metrics['f1@k']:.4f}")
        print(f"  Usuarios avaliados: {metrics['num_users_evaluated']}")
    
    print_separator()
    print("Avaliando metricas de cobertura e diversidade...")
    
    coverage = evaluator.coverage(k=10, sample_users=100)
    diversity = evaluator.diversity(k=10, sample_users=100)
    
    print(f"\nCobertura do catalogo: {coverage:.4f}")
    print(f"  (Proporcao de filmes que aparecem nas recomendacoes)")
    
    print(f"\nDiversidade das recomendacoes: {diversity:.4f}")
    print(f"  (Variedade de filmes recomendados entre usuarios)")
    
    print_separator()
    print("Interpretacao dos resultados:")
    print("\nPrecision@K: Proporcao de itens recomendados que sao relevantes")
    print("  - Valores proximos de 0.1-0.2 sao tipicos em sistemas de recomendacao")
    print("  - Quanto maior, melhor a qualidade das recomendacoes")
    
    print("\nRecall@K: Proporcao de itens relevantes que foram recomendados")
    print("  - Mede a capacidade de encontrar todos os itens relevantes")
    print("  - Trade-off com precision: aumentar K aumenta recall mas diminui precision")
    
    print("\nF1-Score@K: Media harmonica entre Precision e Recall")
    print("  - Balanceia ambas as metricas em um unico valor")
    
    print("\nCobertura: Diversidade do catalogo nas recomendacoes")
    print("  - Valores baixos indicam que sempre recomenda os mesmos filmes")
    print("  - Valores altos indicam exploracao maior do catalogo")
    
    print("\nDiversidade: Variedade entre recomendacoes de diferentes usuarios")
    print("  - Valores proximos de 1.0 = alta personalizacao")
    print("  - Valores proximos de 0.0 = todos recebem as mesmas recomendacoes")
    
    print_separator()


if __name__ == "__main__":
    main()
