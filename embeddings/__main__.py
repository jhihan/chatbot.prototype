import sys
sys.path.append('././')
import constants
import openai
import os
import pandas as pd
from pathlib import Path
import pickle

from doc_emb import compute_doc_embeddings


print("Begin to calculate embedding")

openai.api_key = os.environ.get('openai_api_key')
openai.organization = os.environ.get('openai_organization')

#df_documents = pd.read_json(path_or_buf='./documents/herewithfaqRevised.jsonl', lines=True)
df_documents = pd.read_json(path_or_buf=os.path.join( ".",constants.DIR_DOCUMENTS,constants.document_name), lines=True)
document_embeddings = compute_doc_embeddings(df_documents)

embedding_file_name = Path(constants.document_name).stem + ".pkl"
embedding_file_path = os.path.join(".",".",constants.DIR_DOCUMENTS, constants.DIR_DOCUMENT_EMBEDDINGS, embedding_file_name)
with open(embedding_file_path, 'wb') as outp:
    pickle.dump(document_embeddings, outp, pickle.HIGHEST_PROTOCOL)

print("Finished!")