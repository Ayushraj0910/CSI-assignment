import re
class SQLAgent:
    def __init__(self,db,llm=None): self.db,self.llm=db,llm
    def _fallback_sql(self,q):
        x=q.lower(); filters=[]
        for c in ['Cancer','Obesity','Diabetes','Asthma','Hypertension','Arthritis']:
            if c.lower() in x: filters.append(f"medical_condition = '{c}'")
        for m in ['Paracetamol','Ibuprofen','Aspirin','Penicillin','Lipitor']:
            if m.lower() in x: filters.append(f"medication = '{m}'")
        for a in ['Emergency','Urgent','Elective']:
            if a.lower() in x: filters.append(f"admission_type = '{a}'")
        for i in ['Blue Cross','Medicare','Aetna','UnitedHealthcare','Cigna']:
            if i.lower() in x: filters.append(f"insurance_provider = '{i}'")
        nm=re.search(r'(?:patient|for|about|of)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)',q,re.I)
        if nm and nm.group(1).lower() not in {'patients','cancer','diabetes','obesity','asthma','hypertension','arthritis'}: filters.append(f"name LIKE '%{nm.group(1).strip()}%'")
        where=(' WHERE '+' AND '.join(filters)) if filters else ''
        if any(k in x for k in ['how many','count','number of','total']): return f'SELECT medical_condition, COUNT(*) AS patient_count FROM patients{where} GROUP BY medical_condition LIMIT 50'
        return f'SELECT name, age, gender, blood_type, medical_condition, date_of_admission, doctor, hospital, insurance_provider, billing_amount, room_number, admission_type, discharge_date, medication, test_results FROM patients{where} LIMIT 50'
    def run(self,q):
        sql=None
        if self.llm and self.llm.available:
            p=f'''Translate the question into safe SQLite. Table patients has columns name, age, gender, blood_type, medical_condition, date_of_admission, doctor, hospital, insurance_provider, billing_amount, room_number, admission_type, discharge_date, medication, test_results. SELECT only, LIKE for text, LIMIT 50. Return only SQL. Question: {q}'''
            raw=self.llm.chat([{'role':'system','content':p}],0,500)
            if raw: sql=raw.replace('```sql','').replace('```','').strip().rstrip(';')
        if not sql: sql=self._fallback_sql(q)
        try: rows=self.db.execute(sql)
        except Exception: sql=self._fallback_sql(q); rows=self.db.execute(sql)
        return {'sql':sql,'rows':rows}
    def summarize(self,q,rows):
        if not rows:return 'No matching patient records were found.'
        if not self.llm or not self.llm.available:return f'Found {len(rows)} matching record(s).'
        p=f'Summarize these synthetic hospital database results for staff. Do not invent facts or expose internal IDs. Question: {q}\nResults: {rows[:20]}'
        return self.llm.chat([{'role':'system','content':p}],.1,700) or f'Found {len(rows)} matching record(s).'
