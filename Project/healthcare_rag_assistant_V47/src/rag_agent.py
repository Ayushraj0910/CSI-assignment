from pathlib import Path
import numpy as np
from .config import POLICY_DIR,EMBEDDING_MODEL,TOP_K
class RAGAgent:
    def __init__(self,llm=None): self.llm=llm; self.chunks=[]; self._build()
    def _build(self):
        for path in sorted(Path(POLICY_DIR).glob('*.md')):
            text=path.read_text(encoding='utf-8'); title=text.splitlines()[0].lstrip('# ').strip()
            for i,p in enumerate([x.strip() for x in text.split('\n\n') if len(x.strip())>40]): self.chunks.append({'source':path.name,'title':title,'chunk':i,'text':p})
        texts=[c['text'] for c in self.chunks]
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self.embedder=SentenceTransformer(EMBEDDING_MODEL); v=np.asarray(self.embedder.encode(texts,normalize_embeddings=True,show_progress_bar=False),dtype='float32')
            self.index=faiss.IndexFlatIP(v.shape[1]); self.index.add(v); self.mode='FAISS + SentenceTransformers'
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import faiss
            self.vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2)); v=self.vectorizer.fit_transform(texts).toarray().astype('float32'); v/=np.linalg.norm(v,axis=1,keepdims=True)+1e-8
            self.index=faiss.IndexFlatIP(v.shape[1]); self.index.add(v); self.mode='FAISS + TF-IDF fallback'
    def retrieve(self,q,k=TOP_K):
        if self.mode.startswith('FAISS + Sentence'): v=np.asarray(self.embedder.encode([q],normalize_embeddings=True),dtype='float32')
        else:
            v=self.vectorizer.transform([q]).toarray().astype('float32'); v/=np.linalg.norm(v,axis=1,keepdims=True)+1e-8
        scores,ids=self.index.search(v,min(k,len(self.chunks))); return [(self.chunks[int(i)],float(s)) for s,i in zip(scores[0],ids[0]) if i>=0]
    def answer(self,q):
        got=self.retrieve(q); context='\n\n---\n\n'.join(f"Source: {c['title']}\n{c['text']}" for c,s in got)
        if self.llm and self.llm.available:
            sys='Answer ONLY from the supplied synthetic hospital policy context. If not covered, say so. Cite the source title. Do not invent policy.'
            ans=self.llm.chat([{'role':'system','content':sys},{'role':'user','content':f'Context:\n{context}\n\nQuestion: {q}'}],.1,800)
        else: ans='Retrieved policy context (configure GROQ_API_KEY for generated answers):\n\n'+context
        sources=[{'title':c['title'],'source':c['source'],'score':round(s,3),'text':c['text']} for c,s in got]
        return ans,sources
