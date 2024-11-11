from typing import Dict, Any

from datachat.embeddings.structured_embedding import StructuredEmbedding

class SessionEmbedding(StructuredEmbedding):
    """Semantic embedding for session data"""
    
    def get_text(self, item: Dict[str, Any]) -> str:
        """Convert session data into textual representation"""
        return f"Title: {item['title']}\nSpeaker: {item['nominator']}\nDate: {item['timeslot']}"
    
    def get_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Extract session metadata"""
        return {
            'title': item['title'],
            'speaker': item['nominator'],
            'date': item['timeslot'],
            'type': item['type']
        }
    
    def get_id(self, item: Dict[str, Any]) -> str:
        """Create session identifier"""
        return f"session_{item['id']}"