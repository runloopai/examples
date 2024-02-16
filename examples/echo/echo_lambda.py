import runloop
from pydantic import BaseModel

"""
The echo app implementation does not actually talk to an LLM but validates I/O and scaffolding.
"""


class EchoKv(BaseModel):
    sequence: int = 0


@runloop.function
def echo(echo: str, session: runloop.Session[EchoKv]) -> str:
    session.kv.sequence += 1
    return f"{session.kv.sequence}-{echo}!"
