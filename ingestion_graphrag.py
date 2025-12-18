import os
import glob
import json
import warnings

# Imports pour le Graphe et les Vecteurs
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

warnings.filterwarnings("ignore")

# ==========================================
# 👇 CONFIGURATION 👇
# ==========================================
MY_GROQ_KEY = "votre clef_api_groq_ici"  
MY_NEO4J_URI = "neo4j+s://f0f0d8eb.databases.neo4j.io"
MY_NEO4J_USER = "neo4j"
MY_NEO4J_PASS = "votre_mot_de_passe_neo4j_ici"

os.environ["GROQ_API_KEY"] = MY_GROQ_KEY
os.environ["NEO4J_URI"] = MY_NEO4J_URI
os.environ["NEO4J_USERNAME"] = MY_NEO4J_USER
os.environ["NEO4J_PASSWORD"] = MY_NEO4J_PASS

# 1. Modèle d'Embedding (Transforme le texte en vecteurs)
# C'est un modèle gratuit et performant (all-MiniLM-L6-v2)
print("📥 Chargement du modèle d'embedding...")
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. LLM pour l'extraction d'entités (Groq)
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

def get_files(folder_name):
    # Logique pour trouver le dossier quel que soit l'endroit où on lance le script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # On cherche récursivement ou dans les sous-dossiers courants
    paths_to_check = [
        os.path.join(script_dir, folder_name),
        os.path.join(script_dir, folder_name, folder_name),
        folder_name
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            files = glob.glob(os.path.join(path, "*.txt"))
            if files:
                return files
    return []

def extract_entities(text):
    """ Extrait Symptômes et Maladies via LLM """
    prompt = PromptTemplate(
        template="""
        Analyse ce texte médical. Extrais une liste de SYMPTOMES et une liste de MALADIES.
        Format JSON STRICT: {{"symptomes": ["fièvre", "toux"], "maladies": ["grippe"]}}
        Traduis les termes en Français.
        Texte: {text}
        """,
        input_variables=["text"]
    )
    try:
        chain = prompt | llm
        res = chain.invoke({"text": text})
        # Nettoyage bourrin du JSON pour éviter les erreurs
        json_str = res.content.replace("```json", "").replace("```", "").strip()
        start = json_str.find("{")
        end = json_str.rfind("}") + 1
        return json.loads(json_str[start:end])
    except:
        return {"symptomes": [], "maladies": []}

def build_graph_rag(graph, files):
    print(f"\n🚀 Démarrage de l'ingestion GraphRAG pour {len(files)} fichiers...")
    
    # 1. Nettoyage complet
    graph.query("MATCH (n) DETACH DELETE n")
    
    # 2. Création de l'Index Vectoriel (C'est la clé de l'architecture !)
    # On configure l'index pour des vecteurs de taille 384 (taille du modèle MiniLM)
    try:
        graph.query("DROP INDEX consultation_vector IF EXISTS") # On supprime l'ancien au cas où
        graph.query("""
            CREATE VECTOR INDEX consultation_vector IF NOT EXISTS
            FOR (c:Consultation)
            ON (c.embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: 384,
              `vector.similarity_function`: 'cosine'
            }}
        """)
        print("✅ Index Vectoriel créé dans Neo4j.")
    except Exception as e:
        print(f"⚠️ Info Index: {e}")

    # 3. Traitement des fichiers
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        print(f"   📄 [{i+1}/{len(files)}] Traitement de {filename}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # A. Calcul du Vecteur (Embedding)
        vector = embedding_model.embed_query(content)

        # B. Injection du Noeud Consultation avec son Vecteur
        graph.query("""
            MERGE (c:Consultation {filename: $filename})
            SET c.content = $content,
                c.embedding = $embedding
        """, params={"filename": filename, "content": content, "embedding": vector})

        # C. Extraction et Création des liens Graphiques (Noeuds structurés)
        entities = extract_entities(content)
        
        for sym in entities.get("symptomes", []):
            graph.query("""
                MATCH (c:Consultation {filename: $filename})
                MERGE (s:Symptome {name: toLower($name)})
                MERGE (c)-[:MENTIONNE_SYMPTOME]->(s)
            """, params={"filename": filename, "name": sym})
            
        for mal in entities.get("maladies", []):
            graph.query("""
                MATCH (c:Consultation {filename: $filename})
                MERGE (d:Maladie {name: toLower($name)})
                MERGE (c)-[:MENTIONNE_MALADIE]->(d)
            """, params={"filename": filename, "name": mal})

    print("\n✅ Ingestion terminée ! L'architecture est en place.")

if __name__ == "__main__":
    graph = Neo4jGraph()
    files = get_files("medical_dialogues_50")
    if files:
        build_graph_rag(graph, files)
    else:
        print("❌ Fichiers introuvables. Vérifiez le nom du dossier.")