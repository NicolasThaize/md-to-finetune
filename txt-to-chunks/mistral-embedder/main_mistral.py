#!/usr/bin/env python3
"""
Script de traitement et d'indexation sémantique de documents Markdown
utilisant LlamaIndex avec Mistral AI pour les embeddings.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import (
    Document, 
    VectorStoreIndex, 
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class MarkdownDocumentProcessorMistral:
    """
    Processeur de documents Markdown avec embeddings Mistral AI.
    """
    
    def __init__(self, 
                 markdown_file_path: str,
                 storage_dir: str = "./storage",
                 chunk_size: int = 1024,
                 chunk_overlap: int = 200):
        """
        Initialise le processeur de documents.
        
        Args:
            markdown_file_path: Chemin vers le fichier Markdown
            storage_dir: Répertoire pour stocker l'index
            chunk_size: Taille des chunks
            chunk_overlap: Chevauchement entre chunks
        """
        self.markdown_file_path = Path(markdown_file_path)
        self.storage_dir = Path(storage_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Créer le répertoire de stockage s'il n'existe pas
        self.storage_dir.mkdir(exist_ok=True)
        
        # Configuration LlamaIndex
        self._setup_llamaindex()
        
        # Variables pour l'index
        self.index = None
        self.query_engine = None
        
    def _setup_llamaindex(self):
        """Configure LlamaIndex avec Mistral AI."""
        # Configuration des embeddings Mistral AI
        Settings.embed_model = MistralAIEmbedding(
            model="mistral-embed",
            api_key=os.getenv("MISTRAL_API_KEY")
        )
        
        # Configuration du LLM Mistral AI (optionnel)
        Settings.llm = MistralAI(
            model="mistral-small-latest",
            api_key=os.getenv("MISTRAL_API_KEY"),
            temperature=0.1
        )
        
        logger.info("Configuration LlamaIndex avec Mistral AI terminée")
    
    def load_and_parse_markdown(self) -> List[Document]:
        """
        Charge et parse le fichier Markdown en préservant la structure hiérarchique.
        
        Returns:
            Liste de documents structurés
        """
        logger.info(f"Chargement du fichier: {self.markdown_file_path}")
        
        if not self.markdown_file_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {self.markdown_file_path}")
        
        # Lire le contenu du fichier
        with open(self.markdown_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parser le Markdown en préservant la hiérarchie
        documents = self._parse_markdown_hierarchy(content)
        
        logger.info(f"Document chargé et parsé: {len(documents)} sections trouvées")
        return documents
    
    def _parse_markdown_hierarchy(self, content: str) -> List[Document]:
        """
        Parse le contenu Markdown en préservant la hiérarchie des titres.
        
        Args:
            content: Contenu du fichier Markdown
            
        Returns:
            Liste de documents avec métadonnées de hiérarchie
        """
        documents = []
        lines = content.split('\n')
        current_section = {}
        current_content = []
        
        for i, line in enumerate(lines):
            # Détecter les titres (##, ###, etc.)
            if line.startswith('#'):
                # Sauvegarder la section précédente si elle existe
                if current_section and current_content:
                    doc = self._create_document_from_section(current_section, current_content)
                    if doc:
                        documents.append(doc)
                
                # Commencer une nouvelle section
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                
                current_section = {
                    'title': title,
                    'level': level,
                    'line_number': i + 1
                }
                current_content = [line]
                
            else:
                # Ajouter le contenu à la section courante
                if current_section:
                    current_content.append(line)
        
        # Traiter la dernière section
        if current_section and current_content:
            doc = self._create_document_from_section(current_section, current_content)
            if doc:
                documents.append(doc)
        
        return documents
    
    def _create_document_from_section(self, section: Dict, content: List[str]) -> Optional[Document]:
        """
        Crée un document LlamaIndex à partir d'une section.
        
        Args:
            section: Métadonnées de la section
            content: Contenu de la section
            
        Returns:
            Document LlamaIndex ou None si la section est vide
        """
        full_content = '\n'.join(content).strip()
        
        if not full_content or len(full_content) < 10:  # Ignorer les sections trop courtes
            return None
        
        # Créer le document avec métadonnées
        doc = Document(
            text=full_content,
            metadata={
                'title': section['title'],
                'level': section['level'],
                'line_number': section['line_number'],
                'section_type': 'hierarchy',
                'source': str(self.markdown_file_path)
            }
        )
        
        return doc
    
    def create_chunks_with_hierarchy(self, documents: List[Document]) -> List[Document]:
        """
        Crée des chunks en préservant la hiérarchie des documents.
        
        Args:
            documents: Liste de documents à découper
            
        Returns:
            Liste de chunks avec métadonnées de hiérarchie
        """
        logger.info("Création des chunks avec préservation de la hiérarchie...")
        
        # Utiliser le MarkdownNodeParser pour un découpage intelligent
        node_parser = MarkdownNodeParser()
        
        all_chunks = []
        
        # Traitement par batch pour éviter les rate limits
        batch_size = 2  # Traiter 2 documents à la fois (Mistral AI a des limites)
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            logger.info(f"Traitement du batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
            
            for doc in batch:
                try:
                    # Découper chaque document en chunks
                    nodes = node_parser.get_nodes_from_documents([doc])
                    
                    for node in nodes:
                        # Enrichir les métadonnées du chunk
                        node.metadata.update({
                            'parent_title': doc.metadata.get('title', ''),
                            'parent_level': doc.metadata.get('level', 0),
                            'chunk_type': 'markdown_section'
                        })
                        
                        all_chunks.append(node)
                    
                    # Pause entre les documents pour éviter les rate limits
                    import time
                    time.sleep(2)  # Pause de 2 secondes entre les documents
                    
                except Exception as e:
                    logger.warning(f"Erreur lors du traitement du document {doc.metadata.get('title', 'Sans titre')}: {e}")
                    continue
        
        logger.info(f"Création terminée: {len(all_chunks)} chunks générés")
        return all_chunks
    
    def build_index(self, force_rebuild: bool = False):
        """
        Construit l'index vectoriel pour la recherche sémantique.
        
        Args:
            force_rebuild: Force la reconstruction de l'index
        """
        index_path = self.storage_dir / "index"
        
        # Vérifier si l'index existe déjà
        if index_path.exists() and not force_rebuild:
            logger.info("Chargement de l'index existant...")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(index_path))
                self.index = load_index_from_storage(storage_context)
                logger.info("Index chargé avec succès")
                return
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de l'index: {e}")
                logger.info("Reconstruction de l'index...")
        
        # Construire un nouvel index
        logger.info("Construction d'un nouvel index avec Mistral AI...")
        
        # Charger et parser les documents
        documents = self.load_and_parse_markdown()
        
        # Créer les chunks avec hiérarchie
        chunks = self.create_chunks_with_hierarchy(documents)
        
        # Créer l'index
        self.index = VectorStoreIndex(chunks)
        
        # Sauvegarder l'index
        self.index.storage_context.persist(persist_dir=str(index_path))
        logger.info(f"Index sauvegardé dans: {index_path}")
        
        # Créer le moteur de requête
        self._setup_query_engine()
    
    def _setup_query_engine(self):
        """Configure le moteur de requête pour la recherche sémantique."""
        if self.index is None:
            raise ValueError("L'index doit être construit avant de configurer le moteur de requête")
        
        # Configurer le retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5
        )
        
        # Créer le moteur de requête
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_mode="compact"
        )
        
        logger.info("Moteur de requête configuré")
    
    def search(self, query: str, top_k: int = 5) -> List[NodeWithScore]:
        """
        Effectue une recherche sémantique dans l'index.
        
        Args:
            query: Requête de recherche
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste des nœuds correspondants avec scores
        """
        if self.query_engine is None:
            raise ValueError("Le moteur de requête n'est pas configuré")
        
        logger.info(f"Recherche: '{query}'")
        
        # Effectuer la recherche
        response = self.query_engine.query(query)
        
        return response.source_nodes
    
    def get_detailed_results(self, query: str) -> Dict[str, Any]:
        """
        Obtient des résultats détaillés de recherche avec métadonnées.
        
        Args:
            query: Requête de recherche
            
        Returns:
            Dictionnaire avec les résultats détaillés
        """
        results = self.search(query)
        
        detailed_results = {
            'query': query,
            'total_results': len(results),
            'results': []
        }
        
        for i, node in enumerate(results):
            result = {
                'rank': i + 1,
                'score': getattr(node, 'score', 0.0),
                'content': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                'metadata': {
                    'title': node.metadata.get('title', 'Sans titre'),
                    'parent_title': node.metadata.get('parent_title', ''),
                    'level': node.metadata.get('level', 0),
                    'source': node.metadata.get('source', '')
                }
            }
            detailed_results['results'].append(result)
        
        return detailed_results
    
    def print_search_results(self, query: str):
        """
        Affiche les résultats de recherche de manière formatée.
        
        Args:
            query: Requête de recherche
        """
        results = self.get_detailed_results(query)
        
        print(f"\n🔍 Résultats pour: '{query}'")
        print(f"📊 {results['total_results']} résultats trouvés\n")
        
        for result in results['results']:
            print(f"📄 Résultat #{result['rank']} (Score: {result['score']:.3f})")
            print(f"   📑 Section: {result['metadata']['title']}")
            if result['metadata']['parent_title']:
                print(f"   🏗️  Parent: {result['metadata']['parent_title']}")
            print(f"   📝 Contenu: {result['content']}")
            print("-" * 80)


def main():
    """
    Fonction principale pour démontrer l'utilisation du processeur avec Mistral AI.
    """
    # Configuration
    markdown_file = "/Users/nicolas/Documents/Travail/huggingface/txt-to-chunks/sources/droitadminSmall.md"
    storage_dir = "/Users/nicolas/Documents/Travail/huggingface/txt-to-chunks/storage_mistral"
    
    # Vérifier la présence de la clé API Mistral AI
    if not os.getenv("MISTRAL_API_KEY"):
        print("❌ Erreur: La variable d'environnement MISTRAL_API_KEY n'est pas définie")
        print("   Veuillez définir votre clé API Mistral AI:")
        print("   export MISTRAL_API_KEY='votre_cle_mistral'")
        return
    
    try:
        # Initialiser le processeur
        processor = MarkdownDocumentProcessorMistral(
            markdown_file_path=markdown_file,
            storage_dir=storage_dir
        )
        
        print("🚀 Initialisation du processeur de documents avec Mistral AI...")
        
        # Construire l'index
        processor.build_index(force_rebuild=False)
        
        print("✅ Index construit avec succès!")
        
        # Exemples de recherche
        example_queries = [
            "Qu'est-ce que l'administration?",
            "Définition des personnes publiques",
            "Différence entre établissement public et collectivité territoriale",
            "Service public et externalisation",
            "Organes des personnes publiques"
        ]
        
        print("\n" + "="*80)
        print("🔍 EXEMPLES DE RECHERCHE SÉMANTIQUE (MISTRAL AI)")
        print("="*80)
        
        for query in example_queries:
            processor.print_search_results(query)
            print()
        
        # Interface interactive
        print("\n" + "="*80)
        print("💬 INTERFACE INTERACTIVE (MISTRAL AI)")
        print("="*80)
        print("Tapez vos questions (ou 'quit' pour quitter):")
        
        while True:
            user_query = input("\n❓ Votre question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Au revoir!")
                break
            
            if user_query:
                processor.print_search_results(user_query)
    
    except Exception as e:
        logger.error(f"Erreur: {e}")
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    main()
