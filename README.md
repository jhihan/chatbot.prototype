# Description
In this project, we are developing a prototype chatbot engine which can chat with human and provide the answer on specific 
domain knowledge with Retrieval-Augmented Generation (RAG). In this Github repository, we use the Deutschland-Ticket ( https://int.bahn.de/en/offers/regional/deutschland-ticket ), the subscription ticket for bus and rail offered by Deutsche Bahn, as the domain knowledge

Quick start
=======
Before running this application, please set up the environment variable for openai api key first:
```
export OPENAI_API_KEY = <YOUR_API_KEY>
```
Generate the document_embeddings from the domain knowledge which will be used with RAG approach:
```
python embeddings
```
Run this chatbot application with the prepared document_embeddings:
```
python chatbot.py
```

Update a new use case
==
When there is a new use case, please follow these steps:
1. Upload the knowledge documents in the folder `documents` in the json format and the prompt in `prompts`.
2. Create a config file in `documents/config` in yaml format. For example: 
```
dont_know_answer: "Sorry, given my current knowledge base I cannot answer your request. Let us know at ai@dbahn.com if we should add your request."
document_name: "d-ticket_en.ndjson"
service_name: "Deutsche Bahn"
prompt_file_name: "d-ticket_en_prompt.txt"
```
3. Add the mapping of the usecase name to the config file in `documents/usecase_config_dict.py`.
4. Change the `usecase_name` in `documents/__init__.py`.

Please remember to generate the document_embeddings `python embeddings` before running the chatbot.

If you want to switch between different existing usecases, just change the `usecase_name` in `documents/__init__.py`.

Dependencies
=
The minimum required versions of the respective tools are suggested as followed:
*   Python >= 3.8.1
*   langchain >= 0.0.173
*   openai >= 0.27.6
*   tiktoken >= 0.4.0

Contact
=
This is a simple prototype. I have developed more robust conversational agents in the production environment. For collaboration or freelance opportunities, please feel free to reach out:

* Email: jhihan@gmail.com
* LinkedIn: https://www.linkedin.com/in/jhih-an-you/
