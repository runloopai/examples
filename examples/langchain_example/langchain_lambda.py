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

_HISTORY_KEY = "history"


_logger = logging.getLogger(__name__)
_chat = ChatOpenAI(temperature=0)


@runloop.loop
def langchain_chat(
    metadata: dict[str, str],
    inputs: list[str]
) -> tuple[list[str], dict[str, str]]:
    _logger.debug(f"handle_input metadata={metadata} input={inputs}")
    messages_list = messages_from_dict(
        json.loads(metadata.get(_HISTORY_KEY, "[]")))
    human_message = HumanMessage(content=inputs[0])

    messages = [
        *messages_list,
        human_message
    ]

    ai_response = _chat([_SYSTEM_MSG] + messages)

    _logger.debug(f"got response={ai_response}")

    metadata[_HISTORY_KEY] = json.dumps(
        messages_to_dict(messages + [ai_response]))

    return [ai_response.content], metadata
