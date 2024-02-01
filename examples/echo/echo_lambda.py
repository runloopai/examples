import runloop

"""
The echo app implementation does not actually talk to an LLM but validates I/O and scaffolding.
"""


@runloop.loop
def echo(metadata: dict[str, str], input: list[str]) -> tuple[list[str], dict[str, str]]:
    sequence_num = int(metadata.get("sequence", "0"))
    metadata["sequence"] = str(sequence_num + 1)
    return [f"{sequence_num}-{input[0]}!"], metadata
