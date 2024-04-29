import sys
from typing import Any
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class StreamingCallback(StreamingStdOutCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        sys.stdout.write('\033[96m' + token + '\033[0m')
        sys.stdout.flush()