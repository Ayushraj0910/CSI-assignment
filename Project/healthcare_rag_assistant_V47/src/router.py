class Orchestrator:
    POLICY_TERMS=['policy','policies','guideline','guidelines','hipaa','privacy','disclosure','consent','fmla','leave','billing policy','financial assistance','payment plan','bed allocation','emergency admission policy','medication safety']
    DATA_TERMS=['patient','patients','record','records','doctor','hospital','room','admitted','admission','discharge','medication','diagnosed','diagnosis','blood type','test result','billing amount','insurance','how many','count','list']
    def __init__(self,llm=None): self.llm=llm
    def classify(self,q):
        x=q.lower()
        if any(t in x for t in self.POLICY_TERMS): return 'RAG_AGENT'
        if any(t in x for t in self.DATA_TERMS): return 'SQL_AGENT'
        if self.llm and self.llm.available:
            p='Classify as SQL_AGENT for patient database questions, RAG_AGENT for hospital policy questions, GENERAL otherwise. Return only the label.'
            out=self.llm.chat([{'role':'system','content':p},{'role':'user','content':q}],0,20)
            if out in {'SQL_AGENT','RAG_AGENT','GENERAL'}: return out
        return 'GENERAL'
