"""
Module de traitement des documents et création des embeddings.
Gère le chargement, le découpage et l'indexation des documents PDF.
"""

import logging
from pathlib import Path
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from config import MODEL_NAME, DEVICE, PERSIST_DIRECTORY, PDF_PATHS, CHUNK_SIZE, CHUNK_OVERLAP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handler de documents pour la création d'embeddings.
    
    Charge les documents PDF, les découpe en segments et crée
    une base de données vectorielle pour la recherche sémantique.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        """
        Initialise le processeur avec un modèle d'embedding.

        Args:
            model_name: Nom du modèle d'embedding à utiliser
            device: Périphérique de calcul (CPU/GPU)
        """
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )

    def load_documents(self, pdf_paths: List[str]) -> List[Document]:
        """
        Charge les documents PDF depuis les chemins spécifiés.

        Args:
            pdf_paths: Liste des chemins vers les fichiers PDF

        Returns:
            Liste des documents chargés avec leurs métadonnées

        Logs:
            Warning: Si un fichier n'est pas trouvé
            Error: Si le chargement échoue
        """
        docs = []
        for path in pdf_paths:
            if not Path(path).exists():
                logger.warning(f"File not found: {path}")
                continue
            try:
                loader = PyPDFLoader(path)
                documents = loader.load()
                docs.extend(documents)
                logger.info(f"Successfully loaded: {path}")
            except Exception as e:
                logger.error(f"Error loading {path}: {str(e)}")
        return docs

    def process_documents(self, docs: List[Document]) -> List[Document]:
        """
        Découpe les documents en segments et vérifie les métadonnées.

        Args:
            docs: Liste des documents à traiter

        Returns:
            Liste des segments de documents avec métadonnées
        """
        if not docs:
            logger.warning("No documents to process")
            return []
        
        split_docs = self.text_splitter.split_documents(docs)
        
        # Ensure proper metadata
        for doc in split_docs:
            if not hasattr(doc, 'metadata') or not doc.metadata:
                doc.metadata = {'source': 'Unknown source', 'page': 'Unknown page'}
        
        return split_docs

    def create_vectorstore(self, docs: List[Document], persist_directory: str) -> Chroma:
        """
        Crée et persiste la base de données vectorielle.

        Args:
            docs: Liste des documents à indexer
            persist_directory: Répertoire de stockage

        Returns:
            Instance de la base de données vectorielle Chroma

        Raises:
            ValueError: Si aucun document n'est fourni
        """
        if not docs:
            raise ValueError("No documents provided for vectorstore creation")
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embedding_function,
            persist_directory=persist_directory
        )
        
        return vectorstore

def main():
    try:
        # Initialize processor
        processor = DocumentProcessor()

        # Process documents
        raw_docs = processor.load_documents(PDF_PATHS)
        processed_docs = processor.process_documents(raw_docs)
        vectorstore = processor.create_vectorstore(processed_docs, PERSIST_DIRECTORY)

        # Log results
        logger.info(f"Embeddings created and stored in Chroma vectorstore.")
        logger.info(f"Number of documents: {len(processed_docs)}")
        logger.info(f"Number of embeddings: {vectorstore._collection.count()}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()