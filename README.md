# Corrective RAG System

## Setup
```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
cp .env.example .env         # put your key inside
streamlit run app.py
```

## Pipeline
Upload → Extract → Clean → Split → Embed → ChromaDB →
Retrieve → Grade chunks → (Rewrite query → Re-retrieve) → Generate (context only) → Answer + Sources
