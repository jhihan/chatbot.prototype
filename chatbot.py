"""
This is a Python script that serves as a frontend for a chatbot built with the `langchain` and `llms` libraries.
"""
import sys

# Import necessary libraries

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import os
import pandas as pd
from pathlib import Path
import pickle
from langchain.callbacks.base import BaseCallbackManager

from memory import CustomizedConversationSummaryBufferMemory
from callbacks import StreamingCallback
from utils import lb
from utils import txt_file_2_string

import constants
from embeddings import construct_prompt
from embeddings import get_embedding
from embeddings import chat_history_to_prompt




if __name__ == "__main__":

    prompt = txt_file_2_string(os.path.join(constants.DIR_PROMPTS, constants.prompt_file_name))
    prompt = prompt.format(dont_know_answer=constants.dont_know_answer)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    print(os.path.join(constants.DIR_DOCUMENTS, constants.document_name))
    print( os.path.isfile(os.path.join(constants.DIR_DOCUMENTS, constants.document_name)) )
    print( os.path.exists( constants.DIR_DOCUMENTS ) )
    df_documents = pd.read_json(path_or_buf=os.path.join( constants.DIR_DOCUMENTS,constants.document_name), lines=True)
    embedding_file_name = Path(constants.document_name).stem + ".pkl"
    embedding_file_path = os.path.join(constants.DIR_DOCUMENTS, constants.DIR_DOCUMENT_EMBEDDINGS, embedding_file_name)
    with open( embedding_file_path, 'rb') as inp:
        document_embeddings = pickle.load(inp)

    llm = ChatOpenAI(streaming=True, callback_manager=BaseCallbackManager([StreamingCallback()]),
                     verbose=True, temperature=0, max_tokens=200)

    memory = CustomizedConversationSummaryBufferMemory(llm=llm, max_token_limit=1000, return_messages=True)

    conversation = ConversationChain(
        llm=llm,
        prompt = prompt,
        memory=memory,
        verbose=False
    )

    chat_embedding = {}
    chat_history = []

    functional_string = "NONE"
    print(lb(f"\n{constants.service_name} Service AI: Welcome, I am the {constants.service_name} customer service AI. How can I help you?"))
    print(
        "\nCommands:\n/update : Update embeddings\n/save   : Save embeddings\n/verbose   : Show the whole chat history after each input/output \n/exit   : Terminate this session\n")
    try:
        while True:
            input_string = input("You: ")
            if (input_string == ""):
                continue
            if (input_string.lower() == "/verbose"):
                conversation.verbose = True

                print("\nThe whole chat history will be shown after each chat input/output...\n")
                input_string = input("You: ")
            if (input_string.lower() == "/exit"):
                raise KeyboardInterrupt

            sys.stdout.write(f"\n{constants.service_name} Service AI: ")
            most_similar_chat = chat_history_to_prompt(input_string, chat_embedding, chat_history)
            input_to_chain = construct_prompt(input_string, document_embeddings, df_documents, 'gpt-3.5-turbo', most_similar_chat)

            conversation.predict(input=input_to_chain)

            print("\n")


            history = memory.load_memory_variables({})['history']
            interaction = ""
            if len(history) >= 2:
                interaction = "The Human said:" + history[-2].content + ". And the AI answer:" + history[-1].content
            chat_history.append( interaction )
            embedding = get_embedding( interaction )
            chat_embedding[len(chat_history)] = embedding

    except KeyboardInterrupt:
        print('Bot: Good bye, see you soon! <3')
        if (functional_string.lower() == "show chat"):
            print(memory.load_memory_variables({})['history'])
            for item in memory.load_memory_variables({})['history']:
                print(type(item))
                print(item)
        print()