from typing import Sequence
import logging
import json
import runloop
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    messages_from_dict,
    messages_to_dict
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field

"""
The langchain app implementation.
"""

_SYSTEM_MSG = SystemMessage(
    content="You are a helpful travel agent assisting users to find the best value vacation possible. "
            "Using the tools available to you, you work with the user to identify the best round trip "
            "flight option that aligns to the user's preferences. If you don't know their preferences "
            "or need further information to give a good recommendation, you should ask the user "
            "a question."
)

_logger = logging.getLogger(__name__)
_chat = ChatOpenAI(temperature=0)


class ChatKv(BaseModel):
    messages: Sequence[dict] = []


@runloop.function
def langchain_chat(
    next_message: str,
    session: runloop.Session[ChatKv],
) -> str:
    messages_list = messages_from_dict(session.kv.messages)
    human_message = HumanMessage(content=next_message)

    messages = [
        *messages_list,
        human_message
    ]

    ai_response = _chat([_SYSTEM_MSG] + messages)
    _logger.debug(f"got response={ai_response}")

    session.kv.messages = messages_to_dict(messages + [ai_response])

    return ai_response.content
