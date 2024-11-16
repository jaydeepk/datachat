from typing import Dict, Any

from datachat.document import Document


class SessionDocument(Document):
    """Semantic embedding for session data"""

    def __init__(self, item: Dict[str, Any]) -> None:
        self.item = item

    @property
    def id(self) -> str:
        """Create session identifier"""
        return f"session_{self.item['id']}"

    @property
    def text(self) -> str:
        """Convert session data into textual representation"""
        return f"Title: {self.item['title']}\nSpeaker: {self.item['nominator']}\nDate: {self.item['timeslot']}"

    @property
    def metadata(self) -> Dict[str, Any]:
        """Extract session metadata"""
        return {
            "title": self.item["title"],
            "speaker": self.item["nominator"],
            "date": self.item["timeslot"],
            "type": self.item["type"],
        }
