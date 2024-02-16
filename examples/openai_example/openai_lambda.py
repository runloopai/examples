import json
import logging

import openai
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam

import runloop
from pydantic import BaseModel

"""
The openai app implementation.
"""

_SYSTEM_MSG = {"role": "system",
               "content": "You are a laughably unhelpful travel agent assisting users to find the best value vacation "
                          "possible."
                          "Using the tools available to you, you work with the user to identify the best round trip "
                          "flight option that aligns to the user's preferences. If you don't know their preferences "
                          "or need further information to give a good recommendation, you should ask the user "
                          "a question."}


_HISTORY_KEY = "history"

logger = logging.getLogger(__name__)
_model = "gpt-3.5-turbo-16k"
_client = openai.OpenAI()


class ChatKv(BaseModel):
    messages: list[ChatCompletionMessageParam] = []


@runloop.function
def chat(
    next_message: str,
    session: runloop.Session[ChatKv],
) -> str:
    next_user_line = ChatCompletionUserMessageParam(
        content=next_message, role="user")
    session.kv.messages.append(next_user_line)

    response = _client.chat.completions.create(
        model=_model,
        messages=[_SYSTEM_MSG] + session.kv.messages,
    )

    ai_response_message = response.choices[0].message
    session.kv.messages.append(ChatCompletionAssistantMessageParam(
        content=ai_response_message.content, role="assistant"))

    return ai_response_message.content
