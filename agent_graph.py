import os
import warnings
import sys
import time

# --- IMPORTS ---
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langgraph.prebuilt import create_react_agent
except ImportError:
    sys.exit("❌ Veuillez faire : pip install langgraph")

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

# Initialisation
print("🤖 Initialisation de l'Agent Médical...")
try:
    graph = Neo4jGraph()
    # On garde le modèle 70B pour la performance
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    print(f"❌ Erreur d'initialisation : {e}")
    sys.exit(1)

# ==========================================
# 🛠️ OUTILS
# ==========================================
@tool
def recherche_cas_similaires(query: str) -> str:
    """Recherche dans Neo4j des cas patients, symptômes ou maladies similaires (Base Interne)."""
    print(f"   ⚙️ [Outil: GraphRAG] Recherche : '{query}'")
    try:
        vector = embedding_model.embed_query(query)
        cypher = """
        CALL db.index.vector.queryNodes('consultation_vector', 3, $embedding)
        YIELD node AS c, score
        OPTIONAL MATCH (c)-[:MENTIONNE_SYMPTOME]->(s:Symptome)
        OPTIONAL MATCH (c)-[:MENTIONNE_MALADIE]->(m:Maladie)
        RETURN c.content as content, collect(distinct s.name) as sym, collect(distinct m.name) as mal
        """
        results = graph.query(cypher, params={"embedding": vector})
        if not results: return "Aucun dossier trouvé."
        return "\n".join([f"Dossier: {r['content'][:5000]}... | Sym: {r['sym']} | Mal: {r['mal']}" for r in results])
    except Exception as e:
        return f"Erreur GraphRAG: {e}"

@tool
def statistiques_base_donnees(query: str) -> str:
    """Compte ou fait des statistiques sur la base de données."""
    print(f"   ⚙️ [Outil: Stats] Calcul : '{query}'")
    try:
        prompt = PromptTemplate(
            input_variables=["schema", "question"], 
            template="""
            Generate a Cypher query. Schema: {schema}. Question: {question}.
            Rules: Use 'toLower(n.name) CONTAINS'. Return ONLY the Cypher query.
            """
        )
        chain = GraphCypherQAChain.from_llm(llm=llm, graph=graph, verbose=False, cypher_prompt=prompt, allow_dangerous_requests=True)
        return chain.invoke({"query": query})['result']
    except Exception as e:
        return f"Erreur Stats: {e}"

@tool
def recherche_web_medicale(query: str) -> str:
    """Recherche sur Internet (Infos externes)."""
    print(f"   ⚙️ [Outil: Web] Recherche : '{query}'")
    try:
        return DuckDuckGoSearchRun().invoke(query)
    except Exception as e:
        return f"Erreur Web: {e}"

# ==========================================
# 🧠 FONCTION DE TRAITEMENT
# ==========================================
def run_agent_batch():
    tools = [recherche_cas_similaires, statistiques_base_donnees, recherche_web_medicale]
    agent_app = create_react_agent(llm, tools)

    # --- PROMPT SYSTÈME AMÉLIORÉ (PRIORITÉ INTERNE) ---
    system_prompt = """Tu es un analyste de données médicales expert.
    
    HIÉRARCHIE DES SOURCES (RÈGLE D'OR) :
    1. 🥇 PRIORITÉ : Utilise TOUJOURS 'recherche_cas_similaires' (GraphRAG) en premier.
       Si tu trouves l'information dans un dossier patient interne, base ta réponse UNIQUEMENT là-dessus.
       Cite précisément ce que dit le médecin dans le dossier (ex: "Selon le dossier <03>...").
    
    2. 🥈 SECONDAIRE : Utilise 'recherche_web_medicale' SEULEMENT si la base interne est vide ou muette sur le sujet.
    
    3. Ne dis jamais "Je ne peux pas répondre". Dis "Selon les documents...".
    """

    # --- LISTE DES QUESTIONS A TESTER (BATCH) ---
    # Ajoutez ici toutes les questions que vous voulez tester sur vos fichiers
    questions_test = [
        "Why do I have uncomfortable feeling between the middle of my spine and left shoulder blade?",
        "Quelle est la cause psychologique de la possessivité selon le médecin ?",
        "Pourquoi un taux de hCG qui ne double pas est-il un mauvais signe ?",
        "What is the reason for continuous eye allergy and irritation?",
        "Combien de consultations parlent de problèmes cardiaques ou de palpitations ?"
    ]

    print(f"\n🚀 Démarrage du traitement par lots ({len(questions_test)} questions)...")
    print("="*60)

    for i, question in enumerate(questions_test):
        print(f"\n🔹 QUESTION {i+1}/{len(questions_test)} : {question}")
        print("-" * 30)
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=question)
            ]
            
            # Invocation de l'agent
            result = agent_app.invoke({"messages": messages})
            reponse_finale = result['messages'][-1].content
            
            print(f"\n🤖 RÉPONSE AGENT :\n{reponse_finale}")
            print("="*60)
            
            # Petite pause pour éviter de saturer l'API
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Erreur sur la question '{question}': {e}")

if __name__ == "__main__":
    run_agent_batch()