import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.abspath('backend'))
load_dotenv('backend/.env')

from core.ingestion import load_or_create_index
index = load_or_create_index()
query_engine = index.as_query_engine(similarity_top_k=2)
res = query_engine.query('Đái tháo đường HbA1c')
print('RESPONSE:', res)