import streamlit as st
from rag import RAGSystem
from config import BOOKS_MAP, PERSIST_DIRECTORY, BOOKS_DIR
from generate_embeddings import DocumentProcessor
import os
from pathlib import Path

def check_pdf_files() -> bool:
    """Vérifie l'existence des fichiers PDF nécessaires.

    Returns:
        bool: True si tous les fichiers existent, False sinon.
    
    Note:
        Crée le dossier 'livres' s'il n'existe pas.
        Affiche des messages d'erreur via Streamlit si des fichiers sont manquants.
    """
    if not BOOKS_DIR.exists():
        st.error(f"Le dossier 'livres' n'existe pas. Création du dossier...")
        BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    
    missing_files = []
    for title, path in BOOKS_MAP.items():
        if not Path(path).exists():
            missing_files.append(f"- {title} ({Path(path).name})")
    
    if missing_files:
        st.error("Fichiers PDF manquants :")
        for file in missing_files:
            st.error(file)
        st.error(f"Veuillez placer les fichiers PDF dans le dossier : {BOOKS_DIR}")
        return False
    return True

def create_embeddings_if_needed() -> None:
    """Crée les embeddings des documents si nécessaire.
    
    Cette fonction vérifie d'abord l'existence des fichiers source,
    puis crée les embeddings si le dossier de persistance est vide.
    
    Raises:
        Exception: Si la création des embeddings échoue.
    """
    if not check_pdf_files():
        st.stop()
        
    if not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
        with st.spinner('Première utilisation : création de l\'index des documents...'):
            try:
                processor = DocumentProcessor()
                raw_docs = processor.load_documents(list(BOOKS_MAP.values()))
                processed_docs = processor.process_documents(raw_docs)
                processor.create_vectorstore(processed_docs, PERSIST_DIRECTORY)
                st.success('Index créé avec succès!')
            except Exception as e:
                st.error(f"Erreur lors de la création de l'index: {str(e)}")
                raise e

def initialize_rag() -> RAGSystem:
    """Initialise le système RAG avec création des embeddings si nécessaire.
    
    Returns:
        RAGSystem: Instance initialisée du système RAG.
        
    Note:
        Utilise st.session_state pour éviter les réinitialisations inutiles.
    """
    if 'rag_system' not in st.session_state:
        create_embeddings_if_needed()
        with st.spinner('Initialisation du système RAG...'):
            st.session_state.rag_system = RAGSystem()
    return st.session_state.rag_system

def main():
    st.title("🤖 Assistant IA - Base de Connaissances")
    
    # Sélection des sources
    st.sidebar.header("Configuration")
    selected_books = st.sidebar.multiselect(
        "Sélectionnez les livres à utiliser:",
        options=list(BOOKS_MAP.keys()),
        default=list(BOOKS_MAP.keys())
    )
    
    # Convertir les titres sélectionnés en noms de fichiers
    selected_sources = [BOOKS_MAP[book] for book in selected_books]

    st.markdown("""
    Posez vos questions sur Harry Potter ou Hunger Games!
    L'assistant utilisera sa base de connaissances pour vous répondre.
    """)

    # Initialize RAG system with index creation
    try:
        rag = initialize_rag()
    except Exception as e:
        st.error("Erreur d'initialisation du système")
        st.stop()

    # Query input
    user_query = st.text_area("Votre question:", height=100)
    
    if st.button("Envoyer"):
        if user_query:
            if not selected_books:
                st.warning("Veuillez sélectionner au moins une source.")
                return
                
            with st.spinner('Recherche en cours...'):
                try:
                    # Passer les sources sélectionnées à la requête
                    context, sources = rag.get_relevant_context(user_query, selected_sources=selected_sources)
                    prompt = rag.generate_prompt(user_query, context, sources)
                    answer = rag.generate_answer(prompt)
                    
                    st.markdown("### Réponse:")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Une erreur s'est produite: {str(e)}")
        else:
            st.warning("Veuillez entrer une question.")

    # Afficher des exemples de questions
    with st.expander("Exemples de questions"):
        st.markdown("""
        - Qui est Harry Potter?
        - Décris-moi Poudlard.
        - Qui est Katniss Everdeen?
        - Explique-moi ce que sont les Hunger Games.
        """)

if __name__ == "__main__":
    main()
