
import constants
from langchain.memory import ConversationSummaryBufferMemory
from typing import Any, Dict

class CustomizedConversationSummaryBufferMemory(ConversationSummaryBufferMemory):

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""

        key_input = []
        for k, v in inputs.items():
            if k != self.memory_key:
                key_input.append( k )
        # Before saving the chat interactions, we remove the domain knowledge which is used in the question.
        for key in key_input:
            inputs[key] = inputs[key].split(constants.prefix_Q, 1)[1]

        # We don't save the chat interactions which can not be answered by the chatbot.
        for k, v in outputs.items():
            if constants.dont_know_answer in v:
                return

        super().save_context(inputs, outputs)
        # Prune buffer if it exceeds max token limit
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            self.moving_summary_buffer = self.predict_new_summary(
                pruned_memory, self.moving_summary_buffer
            )
