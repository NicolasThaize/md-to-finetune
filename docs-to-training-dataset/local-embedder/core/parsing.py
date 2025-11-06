from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from llama_index.core import Document


class MarkdownParser(ABC):
    """Interface pour le parsing de documents Markdown."""
    
    @abstractmethod
    def parse(self, content: str) -> List[Document]:
        """Parse le contenu Markdown en documents."""
        raise NotImplementedError


class HierarchicalMarkdownParser(MarkdownParser):
    """Parser Markdown qui préserve la hiérarchie des titres."""
    
    def parse(self, content: str) -> List[Document]:
        documents: List[Document] = []
        lines = content.split('\n')
        current_section: Dict[str, int] = {}
        current_content: List[str] = []
        
        for i, line in enumerate(lines):
            if line.startswith('#'):
                if current_section and current_content:
                    doc = self._create_document_from_section(current_section, current_content)
                    if doc:
                        documents.append(doc)
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                current_section = {'title': title, 'level': level, 'line_number': i + 1}
                current_content = [line]
            else:
                if current_section:
                    current_content.append(line)
        
        if current_section and current_content:
            doc = self._create_document_from_section(current_section, current_content)
            if doc:
                documents.append(doc)
        
        return documents
    
    def _create_document_from_section(self, section: Dict, content: List[str]) -> Optional[Document]:
        full_content = '\n'.join(content).strip()
        if not full_content or len(full_content) < 10:
            return None
        return Document(
            text=full_content,
            metadata={
                'title': section['title'],
                'level': section['level'],
                'line_number': section['line_number'],
                'section_type': 'hierarchy'
            }
        )
