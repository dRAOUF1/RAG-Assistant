# RAG Assistant  📚


> Assistant conversationnel intelligent utilisant RAG (Retrieval-Augmented Generation) pour répondre aux questions sur des œuvres littéraires.

## 🌟 Fonctionnalités

- Recherche sémantique dans plusieurs livres
- Réponses contextualisées avec citations des sources
- Interface utilisateur intuitive
- Sélection flexible des sources



## 🚀 Installation

1. Cloner le dépôt :
```bash
git clone <repository-url>
cd rag
```

2. Créer et activer l'environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

4. Configuration :
   - Placer les PDF dans le dossier `livres/`
   - Configurer la clé API Gemini en ajoutant la variable d'environnement `GEMINI_API_KEY` ou avec Streamlit secrets
   - Ajuster les paramètres selon les besoins

## 💻 Utilisation

1. Préparer les embeddings (première utilisation) :
```bash
python generate_embeddings.py
```

2. Lancer l'application :
```bash
streamlit run app.py
```

3. Accéder à l'interface via le navigateur :
```
http://localhost:8501
```

## 🏗️ Architecture

### Composants Principaux

- **RAGSystem** (`rag.py`) : Moteur de recherche et génération
- **DocumentProcessor** (`generate_embeddings.py`) : Traitement des documents et génération des embeddings
- **Interface** (`app.py`) : Interface utilisateur Streamlit
- **Configuration** (`config.py`) : Paramètres centralisés

### Pipeline de Traitement

1. **Indexation**
   - Chargement des PDFs
   - Segmentation du texte
   - Création des embeddings vectoriels

2. **Recherche**
   - Analyse sémantique
   - Récupération contextuelle
   - Filtrage par source

3. **Génération**
   - Construction du prompt
   - Génération de réponse
   - Formatage et citations
