import os
import glob
import warnings
import time

# --- IMPORTS ---
import langchain
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

warnings.filterwarnings("ignore")

# ==========================================
# 👇 CONFIGURATION DES CLÉS 👇
# ==========================================
# ==========================================
# 👇 CONFIGURATION DES CLÉS 👇
# ==========================================
# (Mettez des fausses valeurs ici pour GitHub)
MY_GROQ_KEY = "gsk_CLE_COMPLETE_ICI"
MY_NEO4J_URI = "neo4j+s://xxxxxxxx.databases.neo4j.io"
MY_NEO4J_USER = "neo4j"
MY_NEO4J_PASS = "MOT_DE_PASSE_ICI"
# ==========================================
# ==========================================

os.environ["GROQ_API_KEY"] = MY_GROQ_KEY
os.environ["NEO4J_URI"] = MY_NEO4J_URI
os.environ["NEO4J_USERNAME"] = MY_NEO4J_USER
os.environ["NEO4J_PASSWORD"] = MY_NEO4J_PASS


def read_data():
    """Lit TOUS les fichiers .txt du dossier Data et gère plusieurs encodages."""
    documents = []
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_path, "Data")

    if not os.path.exists(data_folder):
        print(f"❌ Erreur Dossier : {data_folder}")
        return []

    files = glob.glob(os.path.join(data_folder, "*.txt"))
    print(f"📂 Lecture de {len(files)} fichiers trouvés...")

    for f in files:
        text = None
        for encoding in ["utf-8", "cp1252", "latin-1"]:
            try:
                with open(f, "r", encoding=encoding) as file:
                    text = file.read().strip()
                break  # lecture réussie, on sort de la boucle
            except Exception:
                continue

        if not text:
            print(f"⚠️ Impossible de lire {f} avec les encodages standards.")
            continue

        # Limite facultative : 4000 caractères max par fichier
        short_text = text[:4000]
        documents.append(
            Document(page_content=short_text, metadata={"source": os.path.basename(f)})
        )

    print(f"✅ {len(documents)} fichiers lus avec succès !")
    return documents



def main():
    print("🤖 Chargement des modèles...")

    # 1️⃣ EMBEDDINGS
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 2️⃣ LLM GROQ (Llama 3.3)
    try:
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
    except Exception as e:
        print(f"❌ Erreur Init Groq : {e}")
        return

    llm_transformer = LLMGraphTransformer(llm=llm)

    # 3️⃣ CONNEXION NEO4J
    try:
        graph = Neo4jGraph(
            url=MY_NEO4J_URI,
            username=MY_NEO4J_USER,
            password=MY_NEO4J_PASS
        )
        print("✅ Neo4j Connecté !")
    except Exception as e:
        print(f"❌ Erreur Neo4j : {e}")
        return

    # 4️⃣ LECTURE DES DONNÉES
    all_docs = read_data()
    if not all_docs:
        print("❌ Aucun document trouvé.")
        return

    # --- ÉTAPE 1 : CREATION DU GRAPHE ---
    print(f"\n🚀 ÉTAPE 1 : Construction du Graphe ({len(all_docs)} fichiers)...")
    print("   (Cela va prendre du temps. Ne touchez à rien tant que ce n'est pas fini.)")

    count_ok = 0

    for i, doc in enumerate(all_docs):
        filename = doc.metadata.get("source", "inconnu")
        print(f"\n   📄 [{i+1}/{len(all_docs)}] Traitement de : {filename}...")

        try:
            graph_documents = llm_transformer.convert_to_graph_documents([doc])
            if graph_documents:
                graph.add_graph_documents(graph_documents)
                count_ok += 1
                print(f"      ✅ Ajouté ({len(graph_documents)} structures).")
            else:
                print("      ⚠️ Aucun nœud trouvé.")
            print("      ⏳ Pause 3s...")
            time.sleep(3)
        except Exception as e:
            print(f"      ❌ Erreur : {e}")
            if "429" in str(e) or "413" in str(e):
                print("      🛑 Pause longue (30s)...")
                time.sleep(30)

    print(f"\n✅ FIN ÉTAPE 1 : {count_ok}/{len(all_docs)} fichiers traités avec succès !")

    # --- ÉTAPE 2 : INDEXATION VECTORIELLE ---
    print("\n🚀 ÉTAPE 2 : Indexation Vectorielle...")
    try:
        vector_index = Neo4jVector.from_documents(
            all_docs,
            hf_embeddings,
            url=MY_NEO4J_URI,
            username=MY_NEO4J_USER,
            password=MY_NEO4J_PASS,
            index_name="vector_index"
        )
        print("   -> ✅ Index Vectoriel créé avec succès !")
    except Exception as e:
        print(f"❌ Erreur Vector : {e}")

    # --- ÉTAPE 3 : TEST FINAL (QA GRAPHE) ---
    question = "Quels sont les rôles principaux et leurs interactions ?"
    print(f"\n❓ QUESTION GLOBALE : '{question}'")

    print("\n🔸 Résultat Graphe :")

    CYPHER_GENERATION_TEMPLATE = """
    Tu es un expert Neo4j. Convertis la question en requête Cypher.

    RÈGLES STRICTES :
    1. Utilise uniquement le schéma ci-dessous.
    2. ⚠️ NE JAMAIS retourner la propriété 'embedding'.
    3. Si tu utilises une relation dans le RETURN, tu DOIS lui donner une variable dans le MATCH.
       EXEMPLE : MATCH (a)-[r:TYPE]->(b) RETURN type(r)
    4. Retourne les noms des noeuds et relations.
    5. Limite à 10 résultats maximum.

    Schéma: {schema}
    Question: {question}
    """

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=CYPHER_GENERATION_TEMPLATE
    )

    try:
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True,
            cypher_prompt=cypher_prompt
        )
        result = chain.invoke({"query": question})
        print(f"   🤖 Réponse : {result['result']}")
    except Exception as e:
        print(f"   ❌ Erreur : {e}")


if __name__ == "__main__":
    main()
