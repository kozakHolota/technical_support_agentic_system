from typing import Literal

from langgraph.constants import END

from langgraph.types import Command

from lib.classification import QueryClassification, SupportAgentState
from lib.llm import get_client
from lib.utils import search_query


def classify_query(state: SupportAgentState) -> Command[Literal["search_faq"]]:
    """Classify query and return the appropriate command."""
    classify_llm = get_client(QueryClassification)
    system_prompt = """
    You are a helpful assistant that classifies user queries into the following fields:
    type:
    - problem
    - question
    - complaint
    category:
    - password
    - payment
    - technical
    - account
    - general
    urgency:
    - low
    - medium
    - high
    """
    classification = classify_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["user_query"]},
    ])

    return Command(update={"classification": classification}, goto="search_faq")

def search_faq(state: SupportAgentState) -> Command[Literal["draft_response"]]:
    """Search FAQ and return the answer."""
    found = search_query(state["user_query"])
    return Command(
        update={"search_results": [doc.page_content for doc in found]},
        goto="draft_response",
    )

def draft_response(support_agent_state: SupportAgentState) -> Command[Literal["check_escalation"]]:
    """Draft a response based on the search results."""
    search_results = support_agent_state.get("search_results") or []
    if not search_results:
        return Command(
            update={
                "draft_response": "I'm sorry, I couldn't find relevant information to answer your query. A support agent will assist you shortly.",
                "needs_escalation": True,
            },
            goto="check_escalation",
        )

    resp_llm = get_client(SupportAgentState)
    system_prompt = f"""
    You are a helpful assistant that drafts a response based on the search results.
    User query: {support_agent_state["user_query"]}.
    Search results: {support_agent_state["search_results"]}.
    Classification: {support_agent_state["classification"]}.
    Please gather the necessary information from search_results to a draft response.
    Also, please check if the user query needs escalation.
    Query should be escalated if urgency="high", OR type="complaint", OR next tokens are present: «шахрайство», «судов», «юридичн»
    """

    resp = resp_llm.invoke(system_prompt)
    return Command(update=resp, goto="check_escalation")

def check_escalation(support_agent_state: SupportAgentState) -> Command[Literal[END]]:
    """Check if the user query needs escalation and corrects the draft response if needed."""
    resp_llm = get_client(SupportAgentState)

    needs_escalation = support_agent_state["needs_escalation"]

    system_prompt = f"""You are a corrector of draft responses based on escalation necessity.
    {'This query NEEDS escalation to a human agent.' if needs_escalation else 'No escalation needed.'}
    Draft response: {support_agent_state.get('draft_response', '')}"""

    resp = resp_llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": support_agent_state.get("user_query", "")},
    ])
    return Command(update=resp, goto="__end__")