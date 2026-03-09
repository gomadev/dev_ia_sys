from src.recommender import HybridRecommender


def print_separator():
    print("\n" + "=" * 80 + "\n")


def display_recommendations(recommendations, user_stats):
    print(f"Usuario ID: {user_stats['user_id']}")
    print(f"Total de avaliacoes: {user_stats['total_ratings']}")
    print(f"Media de avaliacao: {user_stats['avg_rating']:.2f}/5.0")
    print(f"Estrategia de recomendacao: {user_stats['recommendation_strategy']}")

    if user_stats['total_ratings'] > 0:
        print("\nFilmes mais bem avaliados pelo usuario:")
        for movie_id, title, rating in user_stats.get('top_rated_movies', []):
            print(f"  - {title} (Rating: {rating:.1f}/5.0)")

    print(f"\nRecomendacoes (Top {len(recommendations)}):")
    for idx, rec in enumerate(recommendations, 1):
        print(f"  {idx}. {rec.title}")
        print(f"     Score: {rec.score:.3f} | Estrategia: {rec.strategy}")
        print(f"     {rec.explanation}")


def main():
    print("Inicializando sistema de recomendacao baseado em Rating...")
    recommender = HybridRecommender(data_dir="data")

    print("Matriz de similaridade construida com sucesso!")
    print_separator()

    test_users = [1, 5, 10, 15, 100]

    for user_id in test_users:
        user_stats = recommender.get_user_stats(user_id)

        if user_stats['total_ratings'] == 0:
            print(f"Usuario {user_id} nao possui avaliacoes.")
            print_separator()
            continue

        recommendations = recommender.recommend(user_id, n=10)
        display_recommendations(recommendations, user_stats)
        print_separator()

    print("\nTeste de usuario novo (sem historico):")
    print("Recomendacoes baseadas nos filmes com melhor Rating:")
    new_user_recommendations = recommender.best_sellers.recommend(n=10)
    for idx, rec in enumerate(new_user_recommendations, 1):
        print(f"  {idx}. {rec.title}")
        print(f"     Score: {rec.score:.3f}")
        print(f"     {rec.explanation}")


if __name__ == "__main__":
    main()
