from typing import Literal, Optional, List
from typing_extensions import TypedDict


class Query(TypedDict):
    """A user classified_query."""
    summary: str

class QueryClassification(TypedDict):
    """A classified_query classification."""
    type: Literal["problem", "question", "complaint"]
    category: Literal["password", "payment", "technical", "account", "general"]
    urgency: Literal["low", "medium", "high"]
    summary: str

class SearchResult(TypedDict):
    """A search result."""
    user_query: str
    classification: Optional[QueryClassification]
    search_results: Optional[List[str]]

class SupportAgentState(TypedDict):
    """Стан агента"""
    user_query: str
    classification: Optional[QueryClassification]
    search_results: Optional[List[str]]
    draft_response: Optional[str]
    needs_escalation: bool