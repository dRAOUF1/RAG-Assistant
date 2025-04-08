"""
Module principal du système RAG (Retrieval-Augmented Generation).
Fournit les fonctionnalités de recherche et génération de réponses.
"""

import os
import signal
import sys
from typing import List, Tuple
from dataclasses import dataclass
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from dotenv import load_dotenv
from config import MODEL_NAME, DEVICE, PERSIST_DIRECTORY, GEMINI_API_KEY

@dataclass
class Source:
    """Représente une source documentaire avec son nom et numéro de page."""
    name: str  # Nom du fichier source
    page: str  # Numéro de page dans le document

class RAGSystem:
    """
    Système de question-réponse basé sur RAG.
    
    Attributs:
        embedding_function: Fonction d'embedding pour la recherche sémantique
        vectorstore: Base de données vectorielle pour le stockage des documents
        model: Modèle génératif pour la production des réponses
    """
    
    def __init__(self, model_name: str = MODEL_NAME, persist_directory: str = PERSIST_DIRECTORY, device: str = DEVICE):
        """
        Initialise le système RAG.
        
        Args:
            model_name: Nom du modèle d'embedding à utiliser
            persist_directory: Répertoire de stockage des embeddings
            device: Périphérique de calcul (CPU/GPU)
        
        Raises:
            ValueError: Si la clé API Gemini n'est pas trouvée
        """
        load_dotenv()  # Load environment variables
        
        self.gemini_key = GEMINI_API_KEY
        if not self.gemini_key:
            raise ValueError("GEMINI_API_KEY not found in configuration")
            
        # Initialize embeddings
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )
        
        # Initialize vector store
        self.vectorstore = Chroma(
            embedding_function=self.embedding_function,
            persist_directory=persist_directory
        )
        
        # Initialize Gemini
        genai.configure(api_key=self.gemini_key)
        self.model = genai.GenerativeModel()

    def get_relevant_context(self, query: str, k: int = 20, selected_sources: List[str] = None) -> Tuple[str, List[Source]]:
        """
        Récupère le contexte pertinent pour la requête donnée.

        Args:
            query: Question de l'utilisateur
            k: Nombre de résultats à retourner
            selected_sources: Liste des sources à utiliser (optionnel)

        Returns:
            Tuple contenant le contexte et la liste des sources utilisées
        """
        search_results = self.vectorstore.similarity_search(query, k=k)
        
        context = ""
        sources = []
        
        for result in search_results:
            if hasattr(result, 'metadata') and result.metadata:
                source_name = result.metadata.get('source', 'Unknown source')
                # Si des sources sont sélectionnées, filtrer les résultats
                if selected_sources and not any(src in source_name for src in selected_sources):
                    continue
                
                page_num = result.metadata.get('page', 'Unknown page')
                context += result.page_content + "\n"
                sources.append(Source(
                    name=os.path.basename(source_name) if source_name != 'Unknown source' else source_name,
                    page=page_num
                ))
        
        return context, sources

    def generate_prompt(self, query: str, context: str, sources: List[Source]) -> str:
        """Generate the RAG prompt with the given context and sources."""
        escaped = context.replace("\n", "\\n")
        sources_text = "\n".join(
            f"Source {i}: {source.name}, Page: {source.page}"
            for i, source in enumerate(sources, 1)
        )
        
        return f"""
        Vous êtes un assistant utile et informatif qui répond aux questions en utilisant le texte du contexte de référence inclus ci-dessous.\
        Assurez-vous de répondre par une phrase complète, en étant exhaustif et en incluant toutes les informations pertinentes.\
        Cependant, vous parlez à un public non technique, alors assurez-vous d'expliquer les concepts complexes de façon simple.\
        Si le contexte n'est pas pertinent pour la réponse, vous pouvez l'ignorer.
        
        TRÈS IMPORTANT: À la fin de votre réponse, citez les sources exactes que vous avez utilisées en indiquant 
        le nom du document et le numéro de page où l'information a été trouvée, en utilisant le format suivant:
        [Source: nom_du_document, Page: numéro_de_page]
        
        Voici les sources disponibles:
        {sources_text}
        
        QUESTION: '{query}'
        CONTEXTE: '{escaped}'
             
        RÉPONSE:
        """

    def generate_answer(self, prompt: str) -> str:
        """Generate an answer using the Gemini model."""
        response = self.model.generate_content(prompt)
        return response.text

    def query(self, user_query: str) -> str:
        """Process a user query and return the generated answer."""
        context, sources = self.get_relevant_context(user_query)
        prompt = self.generate_prompt(user_query, context, sources)
        return self.generate_answer(prompt)

def main():
    # Setup signal handler for graceful exit
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    
    # Initialize RAG system
    rag = RAGSystem()  # No need to pass parameters as they come from config
    
    # Interactive loop
    while True:
        print("-" * 50)
        query = input("Query: ")
        print("-" * 50)
        answer = rag.query(query)
        print(answer)
        print("-" * 50)

if __name__ == "__main__":
    main()
