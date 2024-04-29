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
from langchain.prompts.prompt import PromptTemplate
import os
import pandas as pd
from pathlib import Path
import pickle
from langchain.callbacks.base import BaseCallbackManager

from memory import CustomizedConversationSummaryBufferMemory
from tests import TestPrompt, write_results, test_prompts_files, question_types
from callbacks import StreamingCallback
from utils import print_in_color
from utils import lb
from utils import txt_file_2_string
import yaml

import constants
from embeddings import compute_doc_embeddings
from embeddings import construct_prompt
from embeddings import get_embedding
#from utils import top_document_sections_by_query_similarity
from embeddings import chat_history_to_prompt




if __name__ == "__main__":

    prompt = txt_file_2_string(os.path.join(constants.DIR_PROMPTS, constants.prompt_file_name))
    prompt = prompt.format(dont_know_answer=constants.dont_know_answer)
    print("Before starting this chatbot. Please choose how to setup the prompt.")
    print("\nCommands:\n/1 : system message prompt\n/2 : human message prompt\n/3   : both system and human message prompt\n")
    print("(Default setting is /1)\n")
    input_string = input("Prompt setting: ")
    if (input_string == "/2"):
        prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(prompt +" {input}")
        ])
    elif (input_string == "/3"):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template(prompt +" {input}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

    df_documents = pd.read_json(path_or_buf=os.path.join( constants.DIR_DOCUMENTS,constants.document_name), lines=True)
    embedding_file_name = Path(constants.document_name).stem + ".pkl"
    embedding_file_path = os.path.join(constants.DIR_DOCUMENTS, constants.DIR_DOCUMENT_EMBEDDINGS, embedding_file_name)
    with open( embedding_file_path, 'rb') as inp:
        document_embeddings = pickle.load(inp)

    openai_api_key = os.environ.get('openai_api_key')
    openai_organization = os.environ.get('openai_organization')

    llm = ChatOpenAI(streaming=True, callback_manager=BaseCallbackManager([StreamingCallback()]),
                     verbose=True, temperature=0, max_tokens=200, openai_organization=openai_organization, openai_api_key=openai_api_key)

    #TODO:
    #reenable presence_penalty=-1

    # Create the ConversationChain object with the specified constantsuration
    # memory = ConversationBufferMemory(return_messages=True)
    # memory = CustomizedConversationBufferMemory(return_messages=True)
    # max_token_limit=150 is used to check the functionality of producing summary. In the real application, please set up
    # max_token_limit to a higher value.
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
        "\nCommands:\n/update : Update embeddings\n/save   : Save embeddings\n/tests   : run tests prompts\n/verbose   : Show the whole chat history after each input/output \n/exit   : Terminate this session\n")
    try:
        #verbose = False
        while True:
            input_string = input("You: ")
            if (input_string == ""):
                continue
            if (input_string.lower() == "/update"):
                print("updating embeddings ...")
                document_embeddings = compute_doc_embeddings(df_documents)
                print("finished updating embeddings!\n")
                continue
            if (input_string.lower() == "/save"):
                with open(embedding_file_path, 'wb') as output:
                    # Use pickle.dump to write the data to the file
                    pickle.dump(document_embeddings, output)
                    print("saved")
                continue
            if (input_string.lower() == "/verbose"):
                #conversation.set_verbose(True)
                conversation.verbose = True

                #verbose = True
                print("\nThe whole chat history will be shown after each chat input/output...\n")
                input_string = input("You: ")
            if (input_string.lower() == "/tests"):
                print(
                    "Please choose the evaluation method:\n/1: single GPT model\n/2: majority vote from multiple GPT "
                    "model with non-zero temperature\n (Default setting is /1)")
                evaluation_method = input("Method: ")
                if evaluation_method == "/1":
                    evaluation_method = 1
                elif evaluation_method == "/2":
                    evaluation_method = 2
                else:
                    evaluation_method = 1

                columns_test = ['questions', 'question_types', 'answers', 'approval_condition', 'approval', 'reasons']
                df_test = pd.DataFrame(columns=columns_test)
                for test_prompts_file, question_type in zip(test_prompts_files, question_types):
                    test_prompts = TestPrompt.load_from_json(test_prompts_file)
                    i = 0
                    for test_prompt in test_prompts:
                        i += 1

                        # Clear memory for each tests
                        conversation.memory.clear()

                        print_in_color("\n" + lb(str(i) + ". Question: " + test_prompt.question + "\n"), 'cyan')
                        input_to_chain = construct_prompt(test_prompt.question, document_embeddings, df_documents,
                                                          'gpt-3.5-turbo')
                        answer = conversation.predict(input=input_to_chain)
                        evaluation = test_prompt.evaluate_answer(answer, evaluation_method)
                        print("\n")
                        print_in_color(lb("Approval condition: " + test_prompt.approval_condition), 'cyan')
                        approval = "Yes"
                        reason = ""
                        if evaluation.startswith("Yes"):
                            print_in_color(lb(evaluation), 'green')
                        else:
                            print_in_color(lb(evaluation), 'red')
                            approval = "No"
                            start_char = "No." if evaluation.startswith("No.") else "No"
                            reason = evaluation[len(start_char):]
                        new_row = {columns_test[0]: test_prompt.question,
                                   columns_test[1]: question_type,
                                   columns_test[2]: answer,
                                   columns_test[3]: test_prompt.approval_condition,
                                   columns_test[4]: approval,
                                   columns_test[5]: reason
                                   }
                        df_test = pd.concat([df_test, pd.DataFrame.from_records([new_row])])
                write_results(df_test)
                continue
            if (input_string.lower() == "/exit"):
                raise KeyboardInterrupt

            sys.stdout.write(f"\n{constants.service_name} Service AI: ")
            most_similar_chat = chat_history_to_prompt(input_string, chat_embedding, chat_history)
            input_to_chain = construct_prompt(input_string, document_embeddings, df_documents, 'gpt-3.5-turbo', most_similar_chat)

            #print("\n")
            #print( len(chat_embedding) )
            #if len(chat_embedding) >=1:
            #    chat_similarities = top_document_sections_by_query_similarity(input_string, chat_embedding)
            #    most_similar_chat = chat_history[chat_similarities[1]]
            #    #print( most_similar_chat )

            conversation.predict(input=input_to_chain)

            print("\n")


            history = memory.load_memory_variables({})['history']
            interaction = ""
            if len(history) >= 2:
                interaction = "The Human said:" + history[-2].content + ". And the AI answer:" + history[-1].content
            chat_history.append( interaction )
            embedding = get_embedding( interaction )
            chat_embedding[len(chat_history)] = embedding

            #if verbose is True:
            #    print("\n\nSystemMessage:")
            #    print(prompt.messages[0].prompt.template)
            #    print("Chat history:")
            #    history = memory.load_memory_variables({})['history']
            #    prefix_status = ""
            #    for idx, item in enumerate( history ):
            #        if idx == len(history)-2 or idx == len(history)-1:
            #            prefix_status = "(current)"
            #        print(prefix_status + str(type(item).__name__) + ":" + item.content)
            #    print("")
            #else:
            #    print("\n")

    except KeyboardInterrupt:
        print('Bot: Good bye, see you soon! <3')
        if (functional_string.lower() == "show chat"):
            print(memory.load_memory_variables({})['history'])
            # print( messages_to_dict(memory.load_memory_variables({})['history'].messages) )
            for item in memory.load_memory_variables({})['history']:
                print(type(item))
                print(item)
        print()