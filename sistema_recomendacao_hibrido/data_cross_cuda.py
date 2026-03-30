import os
import numpy as np
import pandas as pd
import faiss
import torch
import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────
# 1. CARREGAR OS DATASETS
# ─────────────────────────────────────────────

def carregar_skoob(path):
        df = pd.read_csv(path)
        df["ISBN_13"] = df["ISBN_13"].astype(str).str.replace(r"[^0-9X]", "", regex=True).str.strip()
        df["ISBN_10"] = df["ISBN_10"].astype(str).str.replace(r"[^0-9X]", "", regex=True).str.strip()
        print(f"Skoob carregado: {len(df)} livros")
        return df


def achar_arquivo(pasta, candidatos):
        for nome in candidatos:
                caminho = os.path.join(pasta, nome)
                if os.path.exists(caminho):
                        return caminho
        raise FileNotFoundError(f"Nenhum desses arquivos encontrado em '{pasta}': {candidatos}")


def carregar_book_crossing(pasta):
        path_ratings = achar_arquivo(pasta, ["Ratings.csv", "BX-Book-Ratings.csv"])
        path_books   = achar_arquivo(pasta, ["Books.csv",   "BX-Books.csv"])

        ratings = pd.read_csv(path_ratings, sep=";", encoding="latin-1", on_bad_lines="skip")
        ratings.columns = ["user_id", "ISBN", "rating_bx"]
        ratings["ISBN"] = ratings["ISBN"].astype(str).str.replace(r"[^0-9X]", "", regex=True).str.strip()
        ratings = ratings[ratings["rating_bx"] > 0]

        livros_bx = pd.read_csv(path_books, sep=";", encoding="latin-1", on_bad_lines="skip", usecols=[0, 1, 2])
        livros_bx.columns = ["ISBN", "titulo_bx", "autor_bx"]
        livros_bx["ISBN"] = livros_bx["ISBN"].astype(str).str.replace(r"[^0-9X]", "", regex=True).str.strip()

        print(f"Book-Crossing carregado: {len(ratings):,} ratings | {ratings['user_id'].nunique():,} usuários")
        return ratings, livros_bx


# ─────────────────────────────────────────────
# 2. CRUZAMENTO POR ISBN
# ─────────────────────────────────────────────

def isbn13_para_10(isbn13):
        try:
                s = str(isbn13).strip()
                if len(s) == 13 and s.startswith("978"):
                        base = s[3:12]
                        total = sum((10 - i) * int(d) for i, d in enumerate(base))
                        check = (11 - (total % 11)) % 11
                        return base + ("X" if check == 10 else str(check))
        except Exception:
                pass
        return None


def cruzar_por_isbn(skoob, ratings_bx):
        colunas_skoob = ["titulo", "autor", "genero", "idioma", "ano",
                                         "paginas", "editora", "rating", "avaliacao",
                                         "leram", "querem_ler", "lendo", "abandonos", "descricao"]

        merge_10 = ratings_bx.merge(
                skoob[["ISBN_10"] + colunas_skoob],
                left_on="ISBN", right_on="ISBN_10", how="inner"
        ).drop(columns=["ISBN_10"])

        isbn_ja_casados = set(merge_10["ISBN"])
        ratings_restantes = ratings_bx[~ratings_bx["ISBN"].isin(isbn_ja_casados)]

        skoob2 = skoob.copy()
        skoob2["ISBN_de_13"] = skoob2["ISBN_13"].apply(isbn13_para_10)

        merge_13 = ratings_restantes.merge(
                skoob2[["ISBN_de_13"] + colunas_skoob],
                left_on="ISBN", right_on="ISBN_de_13", how="inner"
        ).drop(columns=["ISBN_de_13"])

        resultado = pd.concat([merge_10, merge_13], ignore_index=True)

        print(f"\nCruzamento por ISBN:")
        print(f"   Via ISBN-10 : {len(merge_10):,} ratings")
        print(f"   Via ISBN-13 : {len(merge_13):,} ratings")
        print(f"   Total       : {len(resultado):,} ratings")
        print(f"   Livros únicos: {resultado['ISBN'].nunique()}")
        print(f"   Usuários     : {resultado['user_id'].nunique():,}")
        return resultado


# Precisa de GPU NVIDIA com cuda instalado.

def embedding_match_gpu(skoob, ratings_bx, livros_bx, isbn_ja_casados, threshold=0.85):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nEmbedding match na {device.upper()}...")

        # Separar livros sem match ISBN
        colunas_skoob = ["ISBN_10", "titulo", "autor", "genero", "idioma", "ano",
                                         "paginas", "editora", "rating", "avaliacao",
                                         "leram", "querem_ler", "lendo", "abandonos", "descricao"]

        skoob_sem = skoob[~skoob["ISBN_10"].isin(isbn_ja_casados)][colunas_skoob].dropna(subset=["titulo"]).copy()
        bx_sem    = livros_bx[~livros_bx["ISBN"].isin(isbn_ja_casados)].dropna(subset=["titulo_bx"]).copy()

        print(f"   Skoob sem match: {len(skoob_sem):,} livros")
        print(f"   BX sem match   : {len(bx_sem):,} livros")

        # Textos para embedding: "titulo autor"
        textos_skoob = (skoob_sem["titulo"].fillna("") + " " + skoob_sem["autor"].fillna("")).tolist()
        textos_bx    = (bx_sem["titulo_bx"].fillna("") + " " + bx_sem["autor_bx"].fillna("")).tolist()

        # Carregar modelo multilingual (funciona bem pt/en)
        print("\n   Carregando modelo de embeddings...")
        modelo = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

        # Gerar embeddings em batch (GPU faz tudo em paralelo)
        print("   Gerando embeddings Skoob...")
        emb_skoob = modelo.encode(textos_skoob, batch_size=256, show_progress_bar=True,
                                                             convert_to_numpy=True, normalize_embeddings=True)

        print("   Gerando embeddings Book-Crossing...")
        emb_bx = modelo.encode(textos_bx, batch_size=256, show_progress_bar=True,
                                                        convert_to_numpy=True, normalize_embeddings=True)

        # Indexar Skoob no FAISS (busca por similaridade coseno)
        print("\n   Indexando com FAISS...")
        dim = emb_skoob.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner Product = coseno (embeddings normalizados)
        index.add(emb_skoob.astype("float32"))

        # Buscar o vizinho mais próximo de cada livro BX
        print("   Buscando matches...")
        scores, indices = index.search(emb_bx.astype("float32"), k=1)

        # Filtrar pelo threshold de similaridade
        mapeamento = []
        for i, (score, idx) in enumerate(zip(scores[:, 0], indices[:, 0])):
                if score >= threshold:
                        mapeamento.append({
                                "ISBN_BX" : bx_sem.iloc[i]["ISBN"],
                                "ISBN_SK" : skoob_sem.iloc[idx]["ISBN_10"],
                                "score"   : float(score)
                        })

        print(f"\n   Matches encontrados (similaridade ≥ {threshold}): {len(mapeamento)}")

        if not mapeamento:
                print("   Nenhum match encontrado. Tente diminuir o threshold.")
                return pd.DataFrame()

        df_mapa = pd.DataFrame(mapeamento)

        # Expandir ratings com os matches
        ratings_match = ratings_bx[ratings_bx["ISBN"].isin(df_mapa["ISBN_BX"])].copy()
        ratings_match = ratings_match.merge(df_mapa[["ISBN_BX", "ISBN_SK"]], left_on="ISBN", right_on="ISBN_BX")
        ratings_match = ratings_match.merge(skoob_sem, left_on="ISBN_SK", right_on="ISBN_10", how="left")
        ratings_match = ratings_match.drop(columns=["ISBN_BX", "ISBN_SK", "ISBN_10"], errors="ignore")

        print(f"   Ratings adicionados via embedding: {len(ratings_match):,}")
        return ratings_match


# ─────────────────────────────────────────────
# 4. CONSTRUIR A MATRIZ USUÁRIO × LIVRO
# ─────────────────────────────────────────────

def construir_matriz_usuario_livro(df_cruzado, min_aval_user=2, min_aval_livro=2):
        contagem_user  = df_cruzado["user_id"].value_counts()
        contagem_livro = df_cruzado["titulo"].value_counts()

        usuarios_validos = contagem_user[contagem_user >= min_aval_user].index
        livros_validos   = contagem_livro[contagem_livro >= min_aval_livro].index

        df_filtrado = df_cruzado[
                df_cruzado["user_id"].isin(usuarios_validos) &
                df_cruzado["titulo"].isin(livros_validos)
        ].copy()

        n_u = df_filtrado["user_id"].nunique()
        n_l = df_filtrado["titulo"].nunique()

        print(f"\nMatriz Usuário × Livro (filtro: ≥{min_aval_user}/usuário, ≥{min_aval_livro}/livro):")
        print(f"   Usuários : {n_u:,}")
        print(f"   Livros   : {n_l:,}")
        print(f"   Ratings  : {len(df_filtrado):,}")

        if n_u == 0 or n_l == 0:
                print("\nFiltro eliminou todos os dados — usando dataset sem filtro.")
                df_filtrado = df_cruzado.copy()
                n_u = df_filtrado["user_id"].nunique()
                n_l = df_filtrado["titulo"].nunique()

        densidade = (len(df_filtrado) / (n_u * n_l) * 100) if (n_u * n_l) > 0 else 0
        print(f"   Densidade: {densidade:.2f}%")

        matriz = df_filtrado.pivot_table(
                index="user_id", columns="titulo",
                values="rating_bx", aggfunc="mean"
        )
        return df_filtrado, matriz


# ─────────────────────────────────────────────
# 5. FEATURES PARA CONTENT-BASED
# ─────────────────────────────────────────────

def preparar_features_conteudo(df_livros):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import MinMaxScaler
        from scipy.sparse import hstack, csr_matrix

        df = df_livros.drop_duplicates(subset="titulo").copy()
        df["texto"] = (
                df["genero"].fillna("") + " " +
                df["autor"].fillna("") + " " +
                df["editora"].fillna("") + " " +
                df["descricao"].fillna("")
        )

        tfidf = TfidfVectorizer(max_features=500)
        matriz_texto = tfidf.fit_transform(df["texto"])

        features_num = df[["rating", "avaliacao", "leram", "querem_ler", "abandonos"]].fillna(0)
        scaler = MinMaxScaler()
        matriz_num = scaler.fit_transform(features_num)

        features_combinadas = hstack([matriz_texto, csr_matrix(matriz_num)])
        print(f"\nFeatures de conteúdo: {features_combinadas.shape[1]} dimensões para {features_combinadas.shape[0]} livros")
        return df["titulo"].tolist(), features_combinadas


# ─────────────────────────────────────────────
# 6. SALVAR
# ─────────────────────────────────────────────

def salvar_resultados(df_cruzado, matriz, pasta_saida):
        path_hibrido = os.path.join(pasta_saida, "dataset_hibrido.csv")
        path_matriz  = os.path.join(pasta_saida, "matriz_usuario_livro.csv")
        df_cruzado.to_csv(path_hibrido, index=False)
        matriz.to_csv(path_matriz)
        print(f"\nArquivos salvos:")
        print(f"   {path_hibrido}")
        print(f"   {path_matriz}")


# ─────────────────────────────────────────────
# 7. PIPELINE COMPLETO
# ─────────────────────────────────────────────

def pipeline(
        path_skoob,
        pasta_bx,
        pasta_saida=None,
        min_aval_user=2,
        min_aval_livro=2,
        usar_embedding_match=True,
        embedding_threshold=0.85
):
        if pasta_saida is None:
                pasta_saida = pasta_bx

        print("=" * 55)
        print(" PIPELINE: Skoob × Book-Crossing (GPU)")
        print("=" * 55)

        skoob = carregar_skoob(path_skoob)
        ratings_bx, livros_bx = carregar_book_crossing(pasta_bx)

        df_cruzado   = cruzar_por_isbn(skoob, ratings_bx)
        isbn_casados = set(df_cruzado["ISBN"])

        if usar_embedding_match:
                df_emb = embedding_match_gpu(
                        skoob, ratings_bx, livros_bx, isbn_casados,
                        threshold=embedding_threshold
                )
                if not df_emb.empty:
                        df_cruzado = pd.concat([df_cruzado, df_emb], ignore_index=True)
                        print(f"\nTotal após embedding match:")
                        print(f"   Ratings  : {len(df_cruzado):,}")
                        print(f"   Livros   : {df_cruzado['titulo'].nunique()}")
                        print(f"   Usuários : {df_cruzado['user_id'].nunique():,}")

        df_filtrado, matriz = construir_matriz_usuario_livro(df_cruzado, min_aval_user, min_aval_livro)

        titulos, features = preparar_features_conteudo(skoob)

        salvar_resultados(df_filtrado, matriz, pasta_saida)

        print("\nPipeline concluído!")
        print(f"   Livros com features (content) : {len(titulos)}")
        print(f"   Livros na matriz colaborativa : {matriz.shape[1]}")
        print(f"   Usuários na matriz            : {matriz.shape[0]}")

        return df_filtrado, matriz, titulos, features


# ─────────────────────────────────────────────
# EXECUTAR
# ─────────────────────────────────────────────

if __name__ == "__main__":
        BASE = r"D:\\Projetos\\dev_ia_sys\\sistema_recomendacao_hibrido"

        df, matriz, titulos, features = pipeline(
                path_skoob=rf"{BASE}\dados.csv",
                pasta_bx=BASE,
                pasta_saida=BASE,
                min_aval_user=2,
                min_aval_livro=2,
                usar_embedding_match=True,
                embedding_threshold=0.85  
        )