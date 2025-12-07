import os
import glob
import warnings
from PIL import Image  # Utilisé pour lire les métadonnées des images

# --- IMPORTS LANGCHAIN & NEO4J ---
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

warnings.filterwarnings("ignore")

# ==========================================
# 👇 CONFIGURATION OBLIGatoire 👇
# ==========================================
# Remplacez les valeurs ci-dessous par vos propres informations
MY_GROQ_KEY = "REDACTED-GROQ-KEY"
MY_NEO4J_URI = "neo4j+s://f0f0d8eb.databases.neo4j.io"
MY_NEO4J_USER = "neo4j"
MY_NEO4J_PASS = "7CQFEqM4ndgDJjujlmuu6xNZglwcEAtyMPKE-clvWA0"
# ==========================================
# ==========================================

os.environ["GROQ_API_KEY"] = MY_GROQ_KEY
os.environ["NEO4J_URI"] = MY_NEO4J_URI
os.environ["NEO4J_USERNAME"] = MY_NEO4J_USER
os.environ["NEO4J_PASSWORD"] = MY_NEO4J_PASS


# On indique au script de regarder dans brain_tumor_dataset/brain_tumor_dataset
def get_image_metadata(data_folder_name=os.path.join("brain_tumor_dataset", "brain_tumor_dataset")):
    """
    Parcourt les sous-dossiers ('yes', 'no') du dossier de données spécifié
    et extrait les métadonnées de chaque image.
    Gère plusieurs extensions d'images (.jpg, .jpeg, .png).
    """
    metadata_list = []
    # Chemin absolu du script pour trouver le dossier de données de manière fiable
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder_path = os.path.join(script_dir, data_folder_name)

    # --- LIGNE DE DÉBOGAGE AJOUTÉE ---
    print(f"ℹ️  Le script s'attend à trouver le dossier de données ici : {data_folder_path}")
    # ------------------------------------

    if not os.path.exists(data_folder_path):
        print(f"❌ Erreur: Le dossier principal '{data_folder_path}' n'a pas été trouvé.")
        print("   Veuillez vérifier que le dossier 'brain_tumor_dataset' est bien dans le même répertoire que votre script.")
        return []

    labels = ["yes", "no"]
    supported_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif"] # Accepte plusieurs formats

    print(f"📂 Recherche d'images dans les dossiers : {labels}...")

    for label in labels:
        folder_path = os.path.join(data_folder_path, label)
        if not os.path.exists(folder_path):
            # --- LIGNE DE DÉBOGAGE AJOUTÉE ---
            print(f"   -> ⚠️ Échec de la recherche du sous-dossier ici : {folder_path}")
            # ------------------------------------
            continue

        # Boucle sur chaque type d'extension pour ne manquer aucune image
        for ext in supported_extensions:
            for image_path in glob.glob(os.path.join(folder_path, ext)):
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size

                    metadata_list.append({
                        "filename": os.path.basename(image_path),
                        "path": image_path,
                        "label": "tumor" if label == "yes" else "no_tumor",
                        "width": width,
                        "height": height,
                    })
                except Exception as e:
                    print(f"⚠️ Impossible de lire l'image {image_path}: {e}")

    if not metadata_list:
        print("❌ Aucune image n'a été trouvée. Vérifiez vos dossiers et les extensions de fichiers.")
    else:
        print(f"✅ {len(metadata_list)} images trouvées et analysées !")

    return metadata_list


def create_graph_from_metadata(graph, metadata_list):
    """
    Crée les nœuds et relations dans Neo4j à partir de la liste de métadonnées.
    Cette approche est déterministe et n'utilise pas de LLM pour la construction.
    """
    print("\n🚀 ÉTAPE 1 : Construction du Graphe à partir des métadonnées d'images...")

    # Assure l'unicité des labels pour éviter les doublons et améliorer les performances
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Label) REQUIRE l.name IS UNIQUE")
    graph.query("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Image) REQUIRE i.filename IS UNIQUE")


    count_ok = 0
    for i, meta in enumerate(metadata_list):
        print(f"   📄 [{i+1}/{len(metadata_list)}] Traitement de : {meta['filename']}...")
        try:
            # 1. Crée le nœud Label (s'il n'existe pas déjà)
            graph.query(
                "MERGE (l:Label {name: $label_name})",
                params={"label_name": meta["label"]}
            )

            # 2. Crée le nœud Image et la relation vers son Label en une seule requête
            graph.query("""
                MERGE (i:Image {filename: $filename})
                SET i.path = $path, i.width = $width, i.height = $height
                WITH i
                MATCH (l:Label {name: $label_name})
                MERGE (i)-[:IS_CLASSIFIED_AS]->(l)
            """, params={
                "filename": meta["filename"],
                "path": meta["path"],
                "width": meta["width"],
                "height": meta["height"],
                "label_name": meta["label"],
            })
            count_ok += 1
        except Exception as e:
            print(f"      ❌ Erreur lors de l'ajout de {meta['filename']} : {e}")

    print(f"\n✅ FIN ÉTAPE 1 : {count_ok}/{len(metadata_list)} images intégrées au graphe !")


def main():
    """
    Fonction principale qui orchestre la connexion, la lecture des données,
    la création du graphe et l'interrogation.
    """
    print("🤖 Initialisation de l'application...")

    # 1️⃣ CONNEXION NEO4J
    try:
        graph = Neo4jGraph()
        # On nettoie le graphe pour s'assurer de partir de zéro à chaque exécution
        print("🧼 Nettoyage de la base de données Neo4j...")
        graph.query("MATCH (n) DETACH DELETE n")
        print("✅ Neo4j Connecté et base de données nettoyée !")
    except Exception as e:
        print(f"❌ Erreur de connexion à Neo4j : {e}")
        return

    # 2️⃣ LECTURE DES MÉTRADONNÉES DES IMAGES
    all_metadata = get_image_metadata()
    if not all_metadata:
        print("❌ Arrêt du script car aucune donnée d'image n'a été trouvée.")
        return

    # 3️⃣ CRÉATION DU GRAPHE DANS NEO4J
    create_graph_from_metadata(graph, all_metadata)

    # L'étape d'indexation vectorielle n'est pas applicable ici.

    # 4️⃣ LLM GROQ (pour le QA final)
    # 4️⃣ LLM GROQ (pour le QA final)
    # 4️⃣ LLM GROQ (pour le QA final)
    # 4️⃣ LLM GROQ (pour le QA final)
    # 4️⃣ LLM GROQ (pour le QA final)
    try:
        llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
        )

    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation de Groq : {e}")
        return

    # --- ÉTAPE 2 : TEST FINAL (QA GRAPHE) ---
    question = "Compte le nombre total d'images pour chaque label et affiche le résultat."
    print(f"\n🚀 ÉTAPE 2 : Interrogation du Graphe (QA)")
    print(f"❓ QUESTION : '{question}'")


    # Template pour guider le LLM dans la création de la requête Cypher
    CYPHER_GENERATION_TEMPLATE = """
    Tu es un expert en traduction de langage naturel vers des requêtes Neo4j Cypher.
    Ta tâche est de convertir la question de l'utilisateur en une requête Cypher valide
    en te basant exclusivement sur le schéma de la base de données fourni.

    RÈGLES STRICTES :
    1. Utilise UNIQUEMENT les types de nœuds, les relations et les propriétés présents dans le schéma.
       N'invente JAMAIS d'éléments qui ne sont pas dans le schéma.
    2. Analyse la question pour comprendre l'intention (compter, lister, trouver des propriétés, etc.).
    3. La requête doit être syntaxiquement correcte.
    4. Ne retourne PAS la propriété 'embedding' si elle existe.
    5. Sois direct et ne retourne que la requête.

    Schéma de la base de données:
    {schema}

    Question de l'utilisateur:
    {question}

    Requête Cypher :
    """

    cypher_prompt = PromptTemplate(
        input_variables=["schema", "question"],
        template=CYPHER_GENERATION_TEMPLATE
    )

    try:
        # Rafraîchir le schéma pour que le LLM ait la structure la plus à jour
        graph.refresh_schema()

        # Création de la chaîne de QA
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True, # Affiche la requête Cypher générée et le résultat brut
            cypher_prompt=cypher_prompt,
            allow_dangerous_requests=True  # <-- MODIFICATION ICI
        )
        result = chain.invoke({"query": question})
        print("\n✅ Résultat de l'interrogation :")
        print(f"   🤖 Réponse finale : {result['result']}")

    except Exception as e:
        print(f"   ❌ Une erreur est survenue lors de l'interrogation : {e}")


if __name__ == "__main__":
    main()