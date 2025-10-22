#!/usr/bin/env python3
"""
Script de traitement et d'indexation sémantique de documents Markdown
utilisant LlamaIndex avec des embeddings locaux (gratuit et sans limites).
Modèle: distiluse-base-multilingual-cased (optimisé pour le français)
"""

import os
import re
import json
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
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

class MarkdownDocumentProcessorLocal:
    """
    Processeur de documents Markdown avec embeddings locaux (gratuit).
    Utilise le modèle distiluse-base-multilingual-cased optimisé pour le français.
    """
    
    def __init__(self, 
                 markdown_file_path: str,
                 storage_dir: str = "./storage",
                 embedding_model: str = "sentence-transformers/distiluse-base-multilingual-cased",
                 llm_model: str = "mistral:7b-instruct",
                 chunk_size: int = 1024,
                 chunk_overlap: int = 200):
        """
        Initialise le processeur de documents.
        
        Args:
            markdown_file_path: Chemin vers le fichier Markdown
            storage_dir: Répertoire pour stocker l'index
            embedding_model: Modèle d'embedding à utiliser
            chunk_size: Taille des chunks
            chunk_overlap: Chevauchement entre chunks
        """
        self.markdown_file_path = Path(markdown_file_path)
        self.storage_dir = Path(storage_dir)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Créer le répertoire de stockage s'il n'existe pas
        self.storage_dir.mkdir(exist_ok=True)
        
        # Configuration LlamaIndex avec embeddings locaux
        self._setup_llamaindex()
        
        # Variables pour l'index
        self.index = None
        self.query_engine = None
        
    def _setup_llamaindex(self):
        """Configure LlamaIndex avec des embeddings locaux."""
        logger.info(f"Configuration des embeddings locaux: {self.embedding_model}")
        
        # Vérifier la disponibilité du GPU
        gpu_available = self._check_gpu()
        device = "cuda" if gpu_available else "cpu"
        logger.info(f"Device utilisé: {device}")
        
        # Configuration des embeddings locaux
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model,
            device=device,
            # Optimisations pour la performance
            model_kwargs={
                "torch_dtype": "float16" if gpu_available else "float32",
                "trust_remote_code": True
            },
            # Cache des embeddings pour éviter les recalculs
            cache_folder=str(self.storage_dir / "embeddings_cache")
        )
        
        # Configuration du LLM local (Ollama)
        try:
            Settings.llm = Ollama(
                model=self.llm_model,
                temperature=0.1,
                request_timeout=60.0
            )
            logger.info(f"LLM local configuré: {self.llm_model}")
        except Exception as e:
            logger.warning(f"Impossible de configurer le LLM local: {e}")
            logger.info("Le LLM local sera configuré plus tard si nécessaire")
        
        logger.info("Configuration LlamaIndex avec embeddings locaux terminée")
    
    def _check_gpu(self):
        """Vérifie si un GPU est disponible."""
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"GPU détecté: {torch.cuda.get_device_name(0)}")
                logger.info(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                return True
            else:
                logger.info("Aucun GPU détecté, utilisation du CPU")
                return False
        except ImportError:
            logger.warning("PyTorch non installé, utilisation du CPU")
            return False
    
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
        
        for doc in documents:
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
        logger.info("Construction d'un nouvel index avec embeddings locaux...")
        logger.info("⚠️  Première exécution: téléchargement du modèle (peut prendre quelques minutes)")
        
        # Charger et parser les documents
        documents = self.load_and_parse_markdown()
        
        # Créer les chunks avec hiérarchie
        chunks = self.create_chunks_with_hierarchy(documents)
        
        # Créer l'index
        logger.info("Génération des embeddings (cela peut prendre du temps)...")
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
        
        # Créer le moteur de requête (sans LLM)
        self.query_engine = retriever
        
        logger.info("Moteur de requête configuré (recherche pure)")
    
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
        nodes = self.query_engine.retrieve(query)
        
        return nodes
    
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
    
    def generate_qa_pairs(self, chunks: List[Document], output_file: str = "qa_pairs.jsonl"):
        """
        Génère automatiquement des paires question/réponse pour chaque chunk.
        
        Args:
            chunks: Liste des chunks de documents
            output_file: Fichier de sortie JSONL
        """
        logger.info(f"Génération de paires Q/R pour {len(chunks)} chunks...")
        
        # Vérifier que le LLM est configuré
        if not hasattr(Settings, 'llm') or Settings.llm is None:
            logger.error("LLM non configuré. Impossible de générer les paires Q/R.")
            return
        
        qa_pairs = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Traitement du chunk {i+1}/{len(chunks)}")
            
            try:
                # Générer des questions pour ce chunk
                questions = self._generate_questions_for_chunk(chunk)
                
                # Pour chaque question, générer une réponse
                for question in questions:
                    answer = self._generate_answer_for_question(question, chunk)
                    
                    if question and answer:
                        qa_pair = {
                            "messages": [
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer}
                            ]
                        }
                        qa_pairs.append(qa_pair)
                        logger.info(f"✅ Paire Q/R générée: {question[:50]}...")
                
            except Exception as e:
                logger.warning(f"Erreur lors du traitement du chunk {i+1}: {e}")
                continue
        
        # Sauvegarder dans le fichier JSONL
        self._save_qa_pairs_to_jsonl(qa_pairs, output_file)
        
        logger.info(f"✅ Génération terminée: {len(qa_pairs)} paires Q/R sauvegardées dans {output_file}")
        return qa_pairs
    
    def _generate_questions_for_chunk(self, chunk: Document) -> List[str]:
        """
        Génère des questions pour un chunk donné.
        
        Args:
            chunk: Chunk de document
            
        Returns:
            Liste de questions générées
        """
        prompt = f"""
Basé sur le texte suivant, génère 2-3 questions pertinentes qui pourraient être posées par un étudiant en droit administratif.

Texte:
{chunk.text}

Génère des questions qui:
1. Testent la compréhension des concepts clés
2. Demandent des définitions importantes
3. Explorent les différences entre concepts
4. Sont spécifiques au contenu du texte

Format: Une question par ligne, sans numérotation.
"""
        
        try:
            response = Settings.llm.complete(prompt)
            questions_text = str(response)
            
            # Parser les questions (une par ligne)
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
            
            # Filtrer les questions trop courtes ou vides
            questions = [q for q in questions if len(q) > 10 and '?' in q]
            
            return questions[:3]  # Limiter à 3 questions max
            
        except Exception as e:
            logger.warning(f"Erreur lors de la génération de questions: {e}")
            return []
    
    def _generate_answer_for_question(self, question: str, chunk: Document) -> str:
        """
        Génère une réponse pour une question donnée basée sur le chunk.
        
        Args:
            question: Question posée
            chunk: Chunk contenant les informations
            
        Returns:
            Réponse générée
        """
        prompt = f"""
Tu es un assistant spécialisé en droit administratif. Réponds à la question suivante en te basant uniquement sur le texte fourni.

Question: {question}

Texte de référence:
{chunk.text}

Instructions:
1. Réponds uniquement en français
2. Utilise uniquement les informations du texte fourni
3. Sois précis et concis
4. Si la réponse n'est pas dans le texte, dis-le clairement
5. Structure ta réponse de manière claire

Réponse:
"""
        
        try:
            response = Settings.llm.complete(prompt)
            return str(response).strip()
            
        except Exception as e:
            logger.warning(f"Erreur lors de la génération de réponse: {e}")
            return ""
    
    def _save_qa_pairs_to_jsonl(self, qa_pairs: List[Dict], output_file: str):
        """
        Sauvegarde les paires Q/R dans un fichier JSONL.
        
        Args:
            qa_pairs: Liste des paires Q/R
            output_file: Nom du fichier de sortie
        """
        output_path = self.storage_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa_pair in qa_pairs:
                json.dump(qa_pair, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"✅ {len(qa_pairs)} paires Q/R sauvegardées dans {output_path}")
    
    def generate_training_data(self, output_file: str = "training_data.jsonl"):
        """
        Génère un dataset d'entraînement complet à partir du document.
        
        Args:
            output_file: Fichier de sortie pour le dataset
        """
        logger.info("🚀 Génération du dataset d'entraînement...")
        
        # Charger et parser les documents
        documents = self.load_and_parse_markdown()
        
        # Créer les chunks
        chunks = self.create_chunks_with_hierarchy(documents)
        
        # Générer les paires Q/R
        qa_pairs = self.generate_qa_pairs(chunks, output_file)
        
        # Statistiques
        logger.info(f"📊 Dataset généré:")
        logger.info(f"   📄 Documents: {len(documents)}")
        logger.info(f"   📝 Chunks: {len(chunks)}")
        logger.info(f"   ❓ Paires Q/R: {len(qa_pairs)}")
        logger.info(f"   💾 Fichier: {self.storage_dir / output_file}")
        
        return qa_pairs
    
    def get_model_info(self):
        """Affiche les informations sur le modèle utilisé."""
        print("\n🤖 INFORMATIONS DU MODÈLE")
        print("=" * 50)
        print(f"📦 Modèle: {self.embedding_model}")
        print(f"🌍 Multilingue: Oui (optimisé pour le français)")
        print(f"💾 Taille: ~471MB")
        print(f"🧠 Dimension des embeddings: 512")
        print(f"⚡ Device: {'GPU' if self._check_gpu() else 'CPU'}")
        print(f"💰 Coût: Gratuit (local)")


def main():
    """
    Fonction principale pour démontrer l'utilisation du processeur local.
    """
    # Configuration
    markdown_file = "/Users/nicolas/Documents/Travail/huggingface/txt-to-chunks/local-embedder/sources/droitadminSmall.md"
    storage_dir = "/Users/nicolas/Documents/Travail/huggingface/txt-to-chunks/local-embedder/storage"
    
    try:
        # Initialiser le processeur
        processor = MarkdownDocumentProcessorLocal(
            markdown_file_path=markdown_file,
            storage_dir=storage_dir,
            embedding_model="sentence-transformers/distiluse-base-multilingual-cased"
        )
        
        print("🚀 Initialisation du processeur de documents (version locale)...")
        
        # Afficher les informations du modèle
        processor.get_model_info()
        
        # Construire l'index
        processor.build_index(force_rebuild=False)
        
        print("✅ Index construit avec succès (embeddings locaux)!")
        
        # Menu principal
        while True:
            print("\n" + "="*80)
            print("🎯 MENU PRINCIPAL")
            print("="*80)
            print("1. 🔍 Recherche sémantique")
            print("2. 🤖 Générer dataset d'entraînement (Q/R)")
            print("3. 📊 Afficher statistiques")
            print("4. ❌ Quitter")
            
            choice = input("\nChoisissez une option (1-4): ").strip()
            
            if choice == "1":
                # Recherche sémantique
                print("\n" + "="*80)
                print("🔍 RECHERCHE SÉMANTIQUE")
                print("="*80)
                
                # Exemples de recherche
                example_queries = [
                    "Qu'est-ce que l'administration?",
                    "Définition des personnes publiques",
                    "Différence entre établissement public et collectivité territoriale",
                    "Service public et externalisation",
                    "Organes des personnes publiques"
                ]
                
                print("Exemples de requêtes:")
                for i, query in enumerate(example_queries, 1):
                    print(f"  {i}. {query}")
                
                print("\nTapez vos questions (ou 'back' pour retourner au menu):")
                
                while True:
                    user_query = input("\n❓ Votre question: ").strip()
                    
                    if user_query.lower() in ['back', 'retour']:
                        break
                    elif user_query:
                        processor.print_search_results(user_query)
            
            elif choice == "2":
                # Génération de dataset d'entraînement
                print("\n" + "="*80)
                print("🤖 GÉNÉRATION DE DATASET D'ENTRAÎNEMENT")
                print("="*80)
                print("⚠️  Cette opération peut prendre du temps (5-15 minutes)")
                print("📝 Génération de paires question/réponse pour chaque passage...")
                
                confirm = input("\nContinuer? (y/N): ").strip().lower()
                if confirm in ['y', 'yes', 'oui']:
                    try:
                        qa_pairs = processor.generate_training_data("training_data.jsonl")
                        print(f"\n✅ Dataset généré avec succès!")
                        print(f"📊 {len(qa_pairs)} paires Q/R créées")
                        print(f"💾 Fichier: {processor.storage_dir / 'training_data.jsonl'}")
                    except Exception as e:
                        print(f"❌ Erreur lors de la génération: {e}")
                else:
                    print("❌ Génération annulée")
            
            elif choice == "3":
                # Statistiques
                print("\n" + "="*80)
                print("📊 STATISTIQUES")
                print("="*80)
                
                documents = processor.load_and_parse_markdown()
                chunks = processor.create_chunks_with_hierarchy(documents)
                
                print(f"📄 Documents parsés: {len(documents)}")
                print(f"📝 Chunks créés: {len(chunks)}")
                print(f"💾 Répertoire de stockage: {processor.storage_dir}")
                
                # Vérifier si des données d'entraînement existent
                training_file = processor.storage_dir / "training_data.jsonl"
                if training_file.exists():
                    with open(training_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    print(f"🤖 Paires Q/R générées: {len(lines)}")
                else:
                    print("🤖 Aucune donnée d'entraînement générée")
            
            elif choice == "4":
                print("👋 Au revoir!")
                break
            
            else:
                print("❌ Option invalide, veuillez choisir 1-4")
    
    except Exception as e:
        logger.error(f"Erreur: {e}")
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    main()
