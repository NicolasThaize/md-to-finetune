from abc import ABC, abstractmethod
from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser


class ChunkProcessor(ABC):
    """Interface pour le traitement des chunks."""
    
    @abstractmethod
    def process(self, documents: List[Document]) -> List[Document]:
        """Traite les documents en chunks."""
        pass


class MarkdownChunkProcessor(ChunkProcessor):
    """Processeur de chunks spécialisé pour Markdown."""
    
    def __init__(self):
        self.node_parser = MarkdownNodeParser()
    
    def process(self, documents: List[Document]) -> List[Document]:
        """Découpe les documents en chunks sémantiques."""
        all_chunks: List[Document] = []
        for doc in documents:
            nodes = self.node_parser.get_nodes_from_documents([doc])
            for node in nodes:
                node.metadata.update({
                    'parent_title': doc.metadata.get('title', ''),
                    'parent_level': doc.metadata.get('level', 0),
                    'chunk_type': 'markdown_section'
                })
                all_chunks.append(node)
        return all_chunks


