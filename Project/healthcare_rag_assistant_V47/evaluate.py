import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from src.config import DATASET_PATH
from src.database import PatientDatabase
from src.llm import GroqLLM
from src.router import Orchestrator
from src.sql_agent import SQLAgent
from src.rag_agent import RAGAgent
db=PatientDatabase(DATASET_PATH,Path('data/healthcare.db')); llm=GroqLLM(); router=Orchestrator(llm); sql=SQLAgent(db,llm); rag=RAGAgent(llm)
tests=[('How many Cancer patients are there?','SQL_AGENT'),('Show patients prescribed Ibuprofen','SQL_AGENT'),('What is the HIPAA policy for research disclosure?','RAG_AGENT'),('What is the payment plan policy?','RAG_AGENT'),('How are emergency beds assigned?','RAG_AGENT')]
correct=0
for q,e in tests:
    g=router.classify(q); correct+=g==e; print(g,e,q)
print('Routing accuracy:',correct/len(tests))
for q,_ in tests[:2]:
    r=sql.run(q); print(q,r['sql'],len(r['rows']))
for q,_ in tests[2:]:
    _,s=rag.answer(q); print(q,[(x['title'],x['score']) for x in s])
