"""
Configuration globale du système RAG.
Définit les paramètres, chemins et constantes utilisés par l'application.
"""

from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st

# Charger les variables d'environnement (pour développement local)
load_dotenv()

# Répertoires de base
BASE_DIR = Path(__file__).parent
BOOKS_DIR = BASE_DIR / "livres"  # Dossier contenant les livres PDF

# Configuration du modèle d'embeddings
MODEL_NAME = "intfloat/multilingual-e5-large-instruct"  # Modèle multilingue pour les embeddings
DEVICE = "cuda:2"  # Périphérique de calcul (GPU ou CPU)

# Configuration du stockage des vecteurs
PERSIST_DIRECTORY = "books_index"  # Dossier de persistance des embeddings

# Paramètres de traitement des documents
CHUNK_SIZE = 600     # Taille des segments de texte (caractères)
CHUNK_OVERLAP = 100  # Chevauchement entre segments (caractères)

# Documentation des livres sources
BOOKS_MAP = {
    "Harry Potter 1": f"{BOOKS_DIR}/harry-potter-1-lecole-des-sorciers.pdf",     # Premier tome
    "Harry Potter 2": f"{BOOKS_DIR}/harry-potter-2-la-chambre-des-secrets.pdf",  # Deuxième tome
    "Hunger Games": f"{BOOKS_DIR}/CollinsSuzanne-HungerGames-1HungerGames2008.French.ebook_1 (1).pdf"
}

# Chemins des documents PDF
PDF_PATHS = [str(BOOKS_DIR / path) for path in BOOKS_MAP.values()]

# Mapping inverse pour la recherche par nom de fichier
FILES_TO_TITLES = {v: k for k, v in BOOKS_MAP.items()}

# Configuration de l'API Gemini
# Essayer d'abord les secrets Streamlit, puis les variables d'environnement
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY non définie. Utilisez les secrets Streamlit ou le fichier .env")
