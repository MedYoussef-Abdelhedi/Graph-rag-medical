# 🏥 GraphRAG Médical avec Neo4j & Gemini

Ce projet implémente une architecture **GraphRAG (Retrieval-Augmented Generation)** pour l'analyse de documents médicaux.

Contrairement au RAG classique (vectoriel uniquement), ce système construit un **Graphe de Connaissances** (Knowledge Graph) à partir de textes non structurés, permettant de comprendre les relations complexes entre entités (Symptômes, Maladies, Médicaments...).

## 🚀 Architecture

Le projet suit l'approche "Hybrid RAG" (Slide 24 du cours) :
1.  **Extraction d'Entités (NER) :** Utilisation de **Google Gemini 1.5 Flash** pour transformer le texte en nœuds et relations.
2.  **Stockage Graphe :** Base de données **Neo4j (AuraDB Cloud)**.
3.  **Embedding Vectoriel :** Utilisation de **HuggingFace** (`sentence-transformers`) pour la recherche sémantique.
4.  **Orchestration :** Framework **LangChain**.

## 🛠️ Prérequis

- Python 3.10+
- Compte Neo4j AuraDB (Gratuit)
- Clé API Google AI Studio (Gratuit)

## 📦 Installation

1. **Cloner le projet :**
   ```bash
   git clone https://github.com/VOTRE-PSEUDO/Graph-rag-medical.git
   cd Graph-rag-medical
python -m venv venv
# Windows :
venv\Scripts\activate
# Mac/Linux :
source venv/bin/activate
# insatller
pip install -r requirements.txt
# configuration .env
NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=votre_mot_de_passe_neo4j
GOOGLE_API_KEY=votre_cle_api_google_gemini
# run
python main.py
# faire un compte dans neo4j
https://console-preview.neo4j.io