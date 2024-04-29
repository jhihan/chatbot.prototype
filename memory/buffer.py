
import constants
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, get_buffer_string, HumanMessage
from typing import Any, Dict, List

class CustomizedConversationBufferMemory(ConversationBufferMemory):
    """Buffer for storing conversation memory."""

    #def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
    #    """Return history buffer."""
    #    if (len(self.buffer) >= 2 and len(self.buffer[-2].content.split(constants.prefix_Q, 1)) >= 2):
    #        self.buffer[-2] = HumanMessage(content=self.buffer[-2].content.split(constants.prefix_Q, 1)[1])
    #    if len(self.buffer) >= 2:
    #        # todo: replace the following string with some string variable
    #        if constants.dont_know_answer in self.buffer[-1].content:
    #            self.chat_memory.messages = self.buffer[:len(self.buffer) - 2]
    #    return {self.memory_key: self.buffer}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""

        key_input = []
        for k, v in inputs.items():
            if k != self.memory_key:
                key_input.append(k)

        # Before saving the chat interactions, we remove the domain knowledge which is used in the question.
        for key in key_input:
            inputs[key] = inputs[key].split(constants.prefix_Q, 1)[1]

        # We don't save the chat interactions which can not be answered by the chatbot.
        for k, v in outputs.items():
            if constants.dont_know_answer in v:
                return

        super().save_context(inputs, outputs)
