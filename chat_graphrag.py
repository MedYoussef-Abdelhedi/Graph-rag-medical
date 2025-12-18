import os
import warnings
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings

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

# Initialisation des modèles
print("🤖 Initialisation du système GraphRAG...")
graph = Neo4jGraph()
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def graph_rag_search(question):
    """
    C'est ici que la magie de l'architecture opère.
    """
    
    # 1. Vectorisation de la question utilisateur
    question_vector = embedding_model.embed_query(question)
    
    # 2. Requête Hybride (Vecteur + Graphe)
    # On cherche les 3 consultations les plus proches (vector search)
    # ET on récupère les symptômes/maladies connectés (graph traversal)
    cypher_query = """
    CALL db.index.vector.queryNodes('consultation_vector', 3, $embedding)
    YIELD node AS c, score
    
    // Traversée du graphe pour enrichir le contexte
    OPTIONAL MATCH (c)-[:MENTIONNE_SYMPTOME]->(s:Symptome)
    OPTIONAL MATCH (c)-[:MENTIONNE_MALADIE]->(m:Maladie)
    
    RETURN c.filename AS filename, 
           c.content AS content, 
           collect(distinct s.name) AS symptomes, 
           collect(distinct m.name) AS maladies,
           score
    """
    
    results = graph.query(cypher_query, params={"embedding": question_vector})
    return results

def generate_response(question):
    # Etape de Récupération (Retrieval)
    context_data = graph_rag_search(question)
    
    if not context_data:
        return "Je n'ai rien trouvé de pertinent dans la base."
    
    # Construction du Contexte pour le LLM
    context_text = ""
    for doc in context_data:
        context_text += f"""
        --- DOCUMENT PERTINENT (Score: {doc['score']:.2f}) ---
        Source: {doc['filename']}
        Symptômes Identifiés (Graph): {', '.join(doc['symptomes'])}
        Maladies Identifiées (Graph): {', '.join(doc['maladies'])}
        Contenu du dialogue:
        {doc['content']}
        ----------------------------------------------------
        """
    
    # Prompt Augmenté (RAG)
    prompt = f"""
    Tu es un assistant médical expert. 
    Utilise les informations contextuelles ci-dessous (issues d'une recherche vectorielle et graphique) pour répondre à la question.
    
    Si la réponse n'est pas dans le contexte, dis "Je ne sais pas".
    
    CONTEXTE MÉDICAL:
    {context_text}
    
    QUESTION UTILISATEUR: 
    {question}
    
    RÉPONSE:
    """
    
    # Génération
    response = llm.invoke(prompt)
    return response.content

def main():
    print("✅ Système prêt ! Architecture : Vector Search + Graph Traversal.")
    print("   Posez des questions floues ou précises (ex: 'problèmes de comportement', 'hCG').")
    
    while True:
        question = input("\n👤 VOUS : ")
        if question.lower() in ['q', 'quit']: break
        
        try:
            print("   🔍 Analyse Vectorielle & Graphique en cours...")
            answer = generate_response(question)
            print(f"\n🤖 AI : {answer}")
        except Exception as e:
            print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    main()