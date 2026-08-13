# RAG-Based Healthcare Query Assistant

## 1. Project Topic

**RAG-Based Healthcare Query Assistant**

This project is a Python-based healthcare assistant that allows hospital staff to ask questions in plain English.

The system works with:
- Synthetic patient records
- Synthetic hospital policy documents
- General healthcare questions

The main idea is to provide one simple conversational interface where users can access both structured patient information and unstructured hospital policies.

> **Note:** All patient data and policy documents used in this project are synthetic. This project is for academic/demo purposes and is not a medical decision-support system.

---

## 2. Project Pipeline

```text
User Question
      |
      v
Orchestrator Agent
      |
      +----------------------+----------------------+
      |                      |                      |
      v                      v                      v
SQL Agent              RAG Agent             General Assistant
      |                      |                      |
      v                      v                      v
Patient Database       Policy Documents        Groq LLM
      |                      |
      |                 Embeddings
      |                      |
      |                    FAISS
      |                      |
      +----------+-----------+
                 |
                 v
            Final Answer
```

### Query Routing

The **Orchestrator Agent** checks the incoming question and decides which part of the system should handle it.

- Patient record questions → **NLP-to-SQL Agent**
- Hospital policy questions → **RAG Agent**
- General health questions → **General Healthcare Assistant**

---

## 3. Main Components

### Orchestrator Agent

Classifies the user's question and routes it to the appropriate agent.

### NLP-to-SQL Agent

Converts natural-language questions into SQL queries and searches the synthetic patient database.

Example:

```text
How many Diabetes patients are there?
```

The agent generates a database query and presents the result in an understandable format.

### RAG Agent

Handles questions related to hospital policies.

The policy documents are converted into embeddings and stored in a **FAISS** vector index. When a user asks a policy-related question, relevant document sections are retrieved and given to the LLM to generate the answer.

The project contains policy documents covering areas such as:

- HIPAA Privacy
- Billing
- FMLA Leave
- Emergency Admission
- Medication Safety

### General Healthcare Assistant

Handles general health questions such as:

```text
I am having a cold, what should I do?
```

It provides general information and recovery tips without attempting to diagnose the user.

---

## 4. Technologies Used

- **Python** – Main programming language
- **Streamlit** – Web UI
- **Groq / Llama** – Large Language Model
- **FAISS** – Vector similarity search
- **Sentence Transformers** – Text embeddings
- **SQLite** – Patient database
- **Pandas** – Dataset handling
- **Scikit-learn** – Supporting/fallback retrieval
- **Jupyter Notebook** – Project walkthrough

---

## 5. Project Structure

The final project folder is organized as follows:

```text
healthcare_rag_assistant_v4.7/
│
├── README.md
├── How to Run!!!.txt
├── RUN-Healthcare Assistant.bat
├── .env.example
├── app.py
├── evaluate.py
├── requirements.txt
│
├── data/
│   ├── healthcare_dataset.csv
│   └── policies/
│       ├── hipaa_privacy.md
│       ├── billing_policy.md
│       ├── fmla_leave_policy.md
│       ├── emergency_admission_policy.md
│       └── medication_safety_policy.md
│
├── src/
│   ├── config.py
│   ├── database.py
│   ├── llm.py
│   ├── router.py
│   ├── sql_agent.py
│   └── rag_agent.py
│
└── notebooks/
    └── Healthcare_RAG_Walkthrough.ipynb
```

`.venv/` is a local Python virtual environment and is intentionally not included in the project structure above.

---

## 6. Dataset

The project uses:

```text
data/healthcare_dataset.csv
```

The dataset contains synthetic patient information such as:

- Name
- Age
- Gender
- Blood type
- Medical condition
- Date of admission
- Doctor
- Hospital
- Insurance provider
- Billing amount
- Room number
- Admission type
- Discharge date
- Medication
- Test results

No real patient information is used.

---

## 7. How to Run the Project

### Option 1 — One-Click Run

The easiest way to start the application is:

```text
RUN-Healthcare Assistant.bat
```

Double-click this file.

The batch file checks for the project's virtual environment and required packages and then starts the Streamlit application.

The first run may take some time because Python packages need to be installed.

### Option 2 — Run from Folder

Open the project folder in the Terminal

And Run the following commands

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```



## 8. Groq API Key

The application uses Groq for LLM-based responses.

Create an API key from:

https://console.groq.com/keys

Enter the key in the **Groq API Key** field in the application sidebar.

The project also contains:

```text
.env.example
```

which can be used as a template for environment variables.

Example:

```text
GROQ_API_KEY=your_api_key_here
```

Do not upload or share your actual API key publicly.
Refer to my "How To Run!!" txt file for the api key for easy execution...

---

## 9. Example Questions

### Patient Database

```text
How many Diabetes patients are there?
```

```text
Show patients who were prescribed Ibuprofen.
```

```text
What doctor treated Bobby Jackson?
```

```text
How many patients were admitted as Emergency?
```

### Hospital Policies

```text
What does the HIPAA policy say about privacy?
```

```text
What is the hospital billing policy?
```

```text
How are emergency beds assigned?
```

```text
What does the medication safety policy say?
```

### General Health

```text
I am having a cold, what should I do?
```

```text
How long does a common cold usually take to recover?
```

```text
What can help with a cough?
```

---

## 10. RAG Process

The RAG pipeline works as follows:

```text
Hospital Policy Documents
          |
          v
      Text Chunking
          |
          v
       Embeddings
          |
          v
        FAISS
          |
          v
      User Question
          |
          v
    Similarity Search
          |
          v
 Relevant Policy Chunks
          |
          v
       Groq LLM
          |
          v
      Final Answer
```

The retrieved policy content is used as context for the LLM so that policy-related answers are based on the project's documents.

---

## 11. Jupyter Notebook

The project contains:

```text
notebooks/Healthcare_RAG_Walkthrough.ipynb
```

The notebook can be used to understand the dataset, retrieval process and overall project pipeline.

The final application uses **Streamlit** because it provides a proper conversational dashboard and is easier to demonstrate as a working application.

---

## 12. Evaluation

The project contains:

```text
evaluate.py
```

This file can be used for testing the retrieval and answer pipeline.

More evaluation questions can be added later to measure retrieval quality and answer quality.

---

## 13. Limitations

- Patient data is synthetic.
- Hospital policy documents are synthetic.
- General health responses are for educational/general guidance only.
- The application should not be used for medical diagnosis or clinical decisions.
- RAG responses depend on the quality of the available documents.
- Groq API usage depends on the available API limits.
- The system is intended as an academic demonstration.

---

## 14. Future Improvements

Possible improvements include:

- Adding more hospital policy documents
- Improving query classification
- Adding user authentication
- Adding conversation history
- Adding better RAG evaluation
- Adding role-based access to patient records
- Improving response validation
- Deploying the application online

---

## 15. Conclusion

This project demonstrates how **RAG, LLMs, NLP-to-SQL and multi-agent routing** can be combined into a single healthcare application.

The main goal is to make both structured patient information and unstructured hospital policies easier to access through a simple natural-language interface.
