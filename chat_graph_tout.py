import os
import glob
import json
import warnings
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

print("🤖 Initialisation du système GraphRAG...")
graph = Neo4jGraph()
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ==========================================
# 👇 FONCTIONS 👇
# ==========================================
def get_files(folder_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
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
        json_str = res.content.replace("```json", "").replace("```", "").strip()
        start = json_str.find("{")
        end = json_str.rfind("}") + 1
        return json.loads(json_str[start:end])
    except:
        return {"symptomes": [], "maladies": []}

def build_graph_rag(graph, files):
    print(f"\n🚀 Ingestion GraphRAG pour {len(files)} fichiers...")
    
    # 1. Nettoyage complet
    graph.query("MATCH (n) DETACH DELETE n")
    
    # 2. Création de l'Index Vectoriel
    try:
        graph.query("DROP INDEX consultation_vector IF EXISTS")
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

        vector = embedding_model.embed_query(content)
        graph.query("""
            MERGE (c:Consultation {filename: $filename})
            SET c.content = $content,
                c.embedding = $embedding
        """, params={"filename": filename, "content": content, "embedding": vector})

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
    print("\n✅ Ingestion terminée ! Graph prêt pour interrogation.")

def graph_rag_search(question):
    question_vector = embedding_model.embed_query(question)
    cypher_query = """
    CALL db.index.vector.queryNodes('consultation_vector', 3, $embedding)
    YIELD node AS c, score
    OPTIONAL MATCH (c)-[:MENTIONNE_SYMPTOME]->(s:Symptome)
    OPTIONAL MATCH (c)-[:MENTIONNE_MALADIE]->(m:Maladie)
    RETURN c.filename AS filename, 
           c.content AS content, 
           collect(distinct s.name) AS symptomes, 
           collect(distinct m.name) AS maladies,
           score
    """
    return graph.query(cypher_query, params={"embedding": question_vector})

def generate_response(question):
    context_data = graph_rag_search(question)
    if not context_data:
        return "Je n'ai rien trouvé de pertinent dans la base."
    
    context_text = ""
    for doc in context_data:
        context_text += f"""
        --- DOCUMENT PERTINENT (Score: {doc['score']:.2f}) ---
        Source: {doc['filename']}
        Symptômes: {', '.join(doc['symptomes'])}
        Maladies: {', '.join(doc['maladies'])}
        Contenu:
        {doc['content']}
        ----------------------------------------------------
        """
    prompt = f"""
    Tu es un assistant médical expert. 
    Utilise les informations contextuelles ci-dessous pour répondre à la question.
    Si la réponse n'est pas dans le contexte, dis "Je ne sais pas".
    
    CONTEXTE MÉDICAL:
    {context_text}
    
    QUESTION UTILISATEUR: 
    {question}
    
    RÉPONSE:
    """
    response = llm.invoke(prompt)
    return response.content

# ==========================================
# 👇 MAIN 👇
# ==========================================
if __name__ == "__main__":
    files = get_files("medical_dialogues_50")
    if not files:
        print("❌ Fichiers introuvables. Vérifiez le dossier 'medical_dialogues_50'.")
    else:
        # Vérifie si la base est déjà remplie
        check = graph.query("MATCH (c:Consultation) RETURN count(c) AS cnt")
        if check[0]['cnt'] == 0:
            build_graph_rag(graph, files)
        else:
            print("ℹ️ Graph Neo4j déjà rempli, passage direct au mode interrogation.")
        
        print("\n✅ Système prêt ! Posez vos questions (tapez 'q' pour quitter).")
        while True:
            question = input("\n👤 VOUS : ")
            if question.lower() in ['q', 'quit']:
                break
            try:
                print("   🔍 Recherche en cours...")
                answer = generate_response(question)
                print(f"\n🤖 AI : {answer}")
            except Exception as e:
                print(f"❌ Erreur : {e}")
