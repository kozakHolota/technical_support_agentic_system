import os
from typing import TypeVar, TypedDict, Any, Sequence

from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable

from langchain_openai import ChatOpenAI
from pydantic import SecretStr, BaseModel


class _TDBase(TypedDict):
    pass
TD = TypeVar("TD", bound=_TDBase)

def get_client(structured_outut: TD) -> Runnable[PromptValue | str | Sequence[Any], BaseModel | Any]:
    """Create and return an OpenAI client with the provided API key.

    Returns:
        OpenAI: An OpenAI client instance.
    """
    api_key = os.getenv("OPENAI_API_KEY") or ''
    return ChatOpenAI(model="gpt-4o", api_key=SecretStr(api_key), temperature=0.0).with_structured_output(structured_outut)