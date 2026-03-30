"""
=============================================================
 SISTEMA DE RECOMENDAÇÃO HÍBRIDO
 Content-Based + KNN Colaborativo + Slope One
=============================================================

USO:
    python recomendador_hibrido.py

DEPENDÊNCIAS:
    pip install pandas numpy scikit-learn scipy rapidfuzz
"""

import os
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, hstack
from rapidfuzz import process as fuzz_process

warnings.filterwarnings("ignore")

BASE = r"D:\Projetos\dev_ia_sys\sistema_recomendacao_hibrido"


# ═══════════════════════════════════════════════════════════
# 1. CARREGAR DADOS
# ═══════════════════════════════════════════════════════════

def carregar_dados():
        print("Carregando dados...")
        ratings = pd.read_csv(os.path.join(BASE, "dataset_hibrido.csv"))
        skoob   = pd.read_csv(os.path.join(BASE, "dados.csv"))
        print(f"   Ratings : {len(ratings):,}")
        print(f"   Usuários: {ratings['user_id'].nunique():,}")
        print(f"   Livros  : {ratings['titulo'].nunique():,}")
        return ratings, skoob


# ═══════════════════════════════════════════════════════════
# 2. CONTENT-BASED
# ═══════════════════════════════════════════════════════════

class ContentBasedRecomendador:
        def __init__(self, skoob):
                print("\nConstruindo Content-Based...")
                df = skoob.drop_duplicates(subset="titulo").copy()
                df["texto"] = (
                        df["genero"].fillna("") + " " +
                        df["autor"].fillna("") + " " +
                        df["editora"].fillna("") + " " +
                        df["descricao"].fillna("")
                )

                # Guardar TF-IDF treinado para usar no cold start
                self.tfidf   = TfidfVectorizer(max_features=500)
                mat_texto    = self.tfidf.fit_transform(df["texto"])

                cols_num = ["rating", "avaliacao", "leram", "querem_ler", "abandonos"]
                mat_num  = MinMaxScaler().fit_transform(df[cols_num].fillna(0))
                mat_num  = csr_matrix(mat_num)

                self.features    = hstack([mat_texto, mat_num])
                self.titulos     = df["titulo"].tolist()
                self.df          = df.reset_index(drop=True)
                self.titulo_idx  = {t: i for i, t in enumerate(self.titulos)}
                print(f"   {len(self.titulos)} livros indexados | {self.features.shape[1]} features")

        def casar_titulo(self, titulo_digitado, threshold=70):
                """
                Busca fuzzy o título mais próximo na base.
                Retorna (titulo_casado, score) ou (None, 0) se abaixo do threshold.
                """
                resultado = fuzz_process.extractOne(titulo_digitado, self.titulos)
                if resultado and resultado[1] >= threshold:
                        return resultado[0], resultado[1]
                return None, 0

        def recomendar_por_livros(self, livros_avaliados, top_n=10):
                """
                livros_avaliados: dict {titulo: nota (1-10)}
                Faz fuzzy match automático nos títulos antes de buscar.
                """
                # Casar títulos digitados com os da base
                livros_casados = {}
                for titulo_raw, nota in livros_avaliados.items():
                        titulo_casado, score = self.casar_titulo(titulo_raw)
                        if titulo_casado:
                                print(f"   ✓ '{titulo_raw}' → '{titulo_casado}' (similaridade: {score}%)")
                                livros_casados[titulo_casado] = nota
                        else:
                                print(f"   ✗ '{titulo_raw}' não encontrado na base (ignorado)")

                idxs = [self.titulo_idx[t] for t in livros_casados if t in self.titulo_idx]
                if not idxs:
                        return pd.DataFrame()

                notas  = np.array([livros_casados[t] for t in livros_casados if t in self.titulo_idx])
                pesos  = notas / notas.sum()
                perfil = self.features[idxs].toarray()
                perfil = (perfil * pesos[:, None]).sum(axis=0, keepdims=True)

                sims = cosine_similarity(perfil, self.features.toarray())[0]

                ja_vistos = set(idxs)
                resultado = sorted(
                        [(i, sims[i]) for i in range(len(self.titulos)) if i not in ja_vistos],
                        key=lambda x: -x[1]
                )[:top_n]

                return pd.DataFrame([{
                        "titulo":        self.titulos[i],
                        "score_content": round(score, 4),
                        "genero":        self.df.iloc[i]["genero"],
                        "autor":         self.df.iloc[i]["autor"],
                } for i, score in resultado])

        def recomendar_por_generos(self, generos, top_n=10):
                """
                Recomenda para usuário que só informou gêneros.
                Usa o TF-IDF treinado para scores diferenciados (não mais contagem simples).
                """
                texto_perfil  = " ".join(generos)
                perfil_texto  = self.tfidf.transform([texto_perfil])
                zeros_num     = sp.csr_matrix(np.zeros((1, 5)))
                perfil_completo = hstack([perfil_texto, zeros_num])

                sims = cosine_similarity(perfil_completo, self.features)[0]

                resultado = sorted(
                        [(i, sims[i]) for i in range(len(self.titulos))],
                        key=lambda x: -x[1]
                )[:top_n]

                return pd.DataFrame([{
                        "titulo":        self.titulos[i],
                        "score_content": round(score, 4),
                        "genero":        self.df.iloc[i]["genero"],
                        "autor":         self.df.iloc[i]["autor"],
                } for i, score in resultado if score > 0])


# ═══════════════════════════════════════════════════════════
# 3. KNN COLABORATIVO (Item-Based)
# ═══════════════════════════════════════════════════════════

class KNNRecomendador:
        def __init__(self, ratings, k=20):
                print("\nConstruindo KNN Colaborativo (item-based)...")
                self.ratings = ratings
                self.k       = k

                self.matriz = ratings.pivot_table(
                        index="user_id", columns="titulo",
                        values="rating_bx", aggfunc="mean"
                ).fillna(0)

                self.titulos    = list(self.matriz.columns)
                self.titulo_idx = {t: i for i, t in enumerate(self.titulos)}

                mat_sparse  = csr_matrix(self.matriz.values.T)
                self.modelo = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=k+1)
                self.modelo.fit(mat_sparse)
                print(f"   {len(self.titulos)} livros | k={k} vizinhos")

        def recomendar_para_usuario(self, user_id, top_n=10):
                if user_id not in self.matriz.index:
                        return pd.DataFrame()

                user_ratings     = self.matriz.loc[user_id]
                livros_avaliados = user_ratings[user_ratings > 0].index.tolist()
                if not livros_avaliados:
                        return pd.DataFrame()

                scores   = defaultdict(float)
                contagem = defaultdict(int)

                for livro in livros_avaliados:
                        if livro not in self.titulo_idx:
                                continue
                        idx   = self.titulo_idx[livro]
                        vetor = csr_matrix(self.matriz.values.T[idx])
                        dists, indices = self.modelo.kneighbors(vetor, n_neighbors=self.k+1)
                        nota_usuario   = user_ratings[livro]

                        for dist, vizinho_idx in zip(dists[0][1:], indices[0][1:]):
                                vizinho = self.titulos[vizinho_idx]
                                if vizinho not in livros_avaliados:
                                        sim = 1 - dist
                                        scores[vizinho]   += sim * nota_usuario
                                        contagem[vizinho] += 1

                if not scores:
                        return pd.DataFrame()

                resultado = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
                return pd.DataFrame([{
                        "titulo":    titulo,
                        "score_knn": round(score / contagem[titulo], 4)
                } for titulo, score in resultado])


# ═══════════════════════════════════════════════════════════
# 4. SLOPE ONE
# ═══════════════════════════════════════════════════════════

class SlopeOneRecomendador:
        def __init__(self, ratings):
                print("\nConstruindo Slope One...")
                self.matriz  = ratings.pivot_table(
                        index="user_id", columns="titulo",
                        values="rating_bx", aggfunc="mean"
                )
                self.titulos = list(self.matriz.columns)
                self._calcular_desvios()
                print(f"   {len(self.titulos)} livros indexados")

        def _calcular_desvios(self):
                print("   Calculando desvios (pode levar alguns segundos)...")
                mat = self.matriz.values
                n   = mat.shape[1]
                self.desvio   = np.zeros((n, n))
                self.contagem = np.zeros((n, n))

                for u in range(mat.shape[0]):
                        row      = mat[u]
                        avaliados = np.where(~np.isnan(row))[0]
                        for i in avaliados:
                                for j in avaliados:
                                        if i != j:
                                                self.desvio[i][j]   += row[i] - row[j]
                                                self.contagem[i][j] += 1

                mask = self.contagem > 0
                self.desvio[mask] /= self.contagem[mask]
                print("   Desvios calculados!")

        def recomendar_para_usuario(self, user_id, top_n=10):
                if user_id not in self.matriz.index:
                        return pd.DataFrame()

                user_ratings  = self.matriz.loc[user_id]
                avaliados_idx = np.where(~np.isnan(user_ratings.values))[0]
                nao_avaliados = np.where(np.isnan(user_ratings.values))[0]

                if len(avaliados_idx) == 0:
                        return pd.DataFrame()

                scores = {}
                for j in nao_avaliados:
                        num, den = 0.0, 0.0
                        for i in avaliados_idx:
                                c = self.contagem[j][i]
                                if c > 0:
                                        num += (self.desvio[j][i] + user_ratings.iloc[i]) * c
                                        den += c
                        if den > 0:
                                scores[self.titulos[j]] = num / den

                resultado = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
                return pd.DataFrame([{
                        "titulo":         titulo,
                        "score_slopeone": round(score, 4)
                } for titulo, score in resultado])


# ═══════════════════════════════════════════════════════════
# 5. HÍBRIDO
# ═══════════════════════════════════════════════════════════

class SistemaHibrido:
        def __init__(self, content, knn, slopeone,
                                 peso_content=0.3, peso_knn=0.4, peso_slope=0.3):
                self.content   = content
                self.knn       = knn
                self.slopeone  = slopeone
                self.w_content = peso_content
                self.w_knn     = peso_knn
                self.w_slope   = peso_slope

        def recomendar_usuario_existente(self, user_id, top_n=10):
                df_knn   = self.knn.recomendar_para_usuario(user_id, top_n=top_n*2)
                df_slope = self.slopeone.recomendar_para_usuario(user_id, top_n=top_n*2)

                user_row = self.knn.matriz.loc[user_id] if user_id in self.knn.matriz.index else None
                if user_row is not None:
                        avaliados  = {t: r for t, r in user_row.items() if r > 0}
                        df_content = self.content.recomendar_por_livros(avaliados, top_n=top_n*2)
                else:
                        df_content = pd.DataFrame()

                return self._combinar(df_content, df_knn, df_slope, top_n)

        def recomendar_usuario_novo(self, livros_avaliados, generos, top_n=10):
                df_content = pd.DataFrame()

                if livros_avaliados:
                        df_content = self.content.recomendar_por_livros(livros_avaliados, top_n=top_n*2)

                # Sempre combina com gêneros (soma ao perfil de livros se houver)
                if generos:
                        df_genero = self.content.recomendar_por_generos(generos, top_n=top_n*2)
                        if df_content.empty:
                                df_content = df_genero
                        elif not df_genero.empty:
                                # Média dos dois scores para quem aparece nos dois
                                df_content = pd.concat([df_content, df_genero]).groupby("titulo", as_index=False).agg({
                                        "score_content": "mean",
                                        "genero": "first",
                                        "autor":  "first"
                                }).sort_values("score_content", ascending=False).head(top_n*2)

                return self._combinar(df_content, pd.DataFrame(), pd.DataFrame(), top_n)

        def _normalizar(self, series):
                mn, mx = series.min(), series.max()
                if mx == mn:
                        return series * 0 + 1.0
                return (series - mn) / (mx - mn)

        def _combinar(self, df_content, df_knn, df_slope, top_n):
                scores = defaultdict(float)

                if not df_content.empty:
                        df_content = df_content.copy()
                        df_content["score_content"] = self._normalizar(df_content["score_content"])
                        for _, row in df_content.iterrows():
                                scores[row["titulo"]] += self.w_content * row["score_content"]

                if not df_knn.empty:
                        df_knn = df_knn.copy()
                        df_knn["score_knn"] = self._normalizar(df_knn["score_knn"])
                        for _, row in df_knn.iterrows():
                                scores[row["titulo"]] += self.w_knn * row["score_knn"]

                if not df_slope.empty:
                        df_slope = df_slope.copy()
                        df_slope["score_slopeone"] = self._normalizar(df_slope["score_slopeone"])
                        for _, row in df_slope.iterrows():
                                scores[row["titulo"]] += self.w_slope * row["score_slopeone"]

                resultado = sorted(scores.items(), key=lambda x: -x[1])[:top_n]

                idx_map   = self.content.titulo_idx
                df_result = []
                for titulo, score in resultado:
                        row = {"titulo": titulo, "score_hibrido": round(score, 4)}
                        if titulo in idx_map:
                                meta         = self.content.df.iloc[idx_map[titulo]]
                                row["autor"] = meta.get("autor", "")
                                row["genero"]= meta.get("genero", "")
                        df_result.append(row)

                return pd.DataFrame(df_result)


# ═══════════════════════════════════════════════════════════
# 6. INTERFACE TERMINAL
# ═══════════════════════════════════════════════════════════

def exibir_recomendacoes(df, titulo="Recomendações"):
        print(f"\n{'═'*55}")
        print(f" {titulo}")
        print(f"{'═'*55}")
        if df.empty:
                print("  Nenhuma recomendação encontrada.")
                return
        for i, row in df.reset_index(drop=True).iterrows():
                score  = row.get("score_hibrido", row.get("score_content", ""))
                autor  = row.get("autor", "")
                generos_raw = str(row.get("genero", ""))
                # Limpar gênero longo (pegar só até 80 chars)
                genero = generos_raw[:80] + "..." if len(generos_raw) > 80 else generos_raw
                print(f"  {i+1:2}. {row['titulo']}")
                print(f"      Autor: {autor} | Score: {score}")
                print(f"      Gênero: {genero}")


def modo_usuario_existente(sistema, knn):
        print("\n── Modo: Usuário Existente ──")
        print("Exemplos de user_id válidos:")
        for uid in list(knn.matriz.index[:5]):
                print(f"   {uid}")

        user_id = input("\nDigite o user_id: ").strip()
        try:
                user_id = int(user_id)
        except ValueError:
                pass

        top_n = int(input("Quantas recomendações? [10]: ").strip() or "10")
        df    = sistema.recomendar_usuario_existente(user_id, top_n=top_n)
        exibir_recomendacoes(df, f"Top {top_n} para usuário {user_id}")


def modo_usuario_novo(sistema):
        print("\n── Modo: Usuário Novo (Cold Start) ──")
        print("\nDigite seus gêneros favoritos separados por vírgula:")
        print("Exemplos: Romance, Ficção Científica, Terror, Fantasia, Autoajuda")
        generos_input = input("> ").strip()
        generos = [g.strip() for g in generos_input.split(",") if g.strip()]

        livros_avaliados = {}
        print("\nDigite títulos de livros que você já leu e sua nota (Enter em branco para pular):")
        print("(O sistema busca o título mais próximo automaticamente)")
        while True:
                titulo = input("  Título: ").strip()
                if not titulo:
                        break
                try:
                        nota = float(input(f"  Nota para '{titulo}' (1-10): ").strip())
                        livros_avaliados[titulo] = nota
                except ValueError:
                        print("  Nota inválida, pulando.")

        top_n = int(input("\nQuantas recomendações? [10]: ").strip() or "10")
        df    = sistema.recomendar_usuario_novo(livros_avaliados, generos, top_n=top_n)
        exibir_recomendacoes(df, f"Top {top_n} recomendações personalizadas")


def menu_principal(sistema, knn):
        print("\n" + "═"*55)
        print(" SISTEMA DE RECOMENDAÇÃO HÍBRIDO DE LIVROS")
        print("    Content-Based + KNN + Slope One")
        print("═"*55)

        while True:
                print("\nO que deseja fazer?")
                print("  1. History-based")
                print("  2. CVold STart")
                print("  0. Sair")

                op = input("\nEscolha: ").strip()
                if op == "1":
                        modo_usuario_existente(sistema, knn)
                elif op == "2":
                        modo_usuario_novo(sistema)
                elif op == "0":
                        print("\nfim")
                        break
                else:
                        print("Opção inválida.")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
        ratings, skoob = carregar_dados()

        content  = ContentBasedRecomendador(skoob)
        knn      = KNNRecomendador(ratings, k=20)
        slopeone = SlopeOneRecomendador(ratings)

        sistema = SistemaHibrido(
                content, knn, slopeone,
                peso_content=0.3,
                peso_knn=0.4,
                peso_slope=0.3
        )

        menu_principal(sistema, knn)