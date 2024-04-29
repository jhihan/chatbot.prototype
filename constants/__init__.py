import os
import yaml

from constants.usecase_config_dict import usecase_config_dict

usecase_name = "d-ticket_en"

prefix_Q = "Q: "

DIR_CONSTANTS = "constants"
DIR_CONFIG = "config"
DIR_PROMPTS = "prompts"
DIR_DOCUMENTS = "documents"
DIR_DOCUMENT_EMBEDDINGS = "document_embeddings"

usecase_config = usecase_config_dict[usecase_name]
usecase_config_file = os.path.join( DIR_CONSTANTS, DIR_CONFIG, usecase_config )
with open(usecase_config_file, 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.Loader)

dont_know_answer = config['dont_know_answer']
document_name = config['document_name']
service_name = config['service_name']
prompt_file_name = config['prompt_file_name']