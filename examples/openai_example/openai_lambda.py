import json
import logging

import openai
import runloop

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


@runloop.loop
def chat(
    metadata: dict[str, str],
    inputs: list[str]
) -> tuple[list[str], dict[str, str]]:
    logger.debug(f"handle_input metadata={metadata} input={inputs}")
    next_user_line = {"role": "user", "content": inputs[0]}
    existing_non_system_messages = json.loads(metadata.get(_HISTORY_KEY, "[]"))

    messages_to_process = existing_non_system_messages + [next_user_line]
    response = _client.chat.completions.create(
        model=_model,
        messages=[_SYSTEM_MSG] + messages_to_process,
    )

    logger.debug(f"got response={response}")
    # TODO: determine how to propagate errors
    ai_response_message = response.choices[0].message

    metadata[_HISTORY_KEY] = json.dumps(
        messages_to_process + [ai_response_message.model_dump_json()])

    return [ai_response_message.content], metadata
