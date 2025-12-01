import os
import pandas as pd
import torch
import time
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==========================================
# 0. Hardware & Configuration
# ==========================================
print("=== Hardware Check ===")
print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Available: Yes")
    print(f"GPU Count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA Available: No (WARNING: This will be slow on CPU)")
print("======================\n")

# Configuration
# "google/flan-t5-large" is robust and does not require a gated token.
# If you want to use "meta-llama/Llama-3.2-3B-Instruct", ensure HF_TOKEN is set.
MODEL_ID = "google/flan-t5-large" 
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Get HF Token from environment variable for gated models
HF_TOKEN = os.getenv('HF_TOKEN') 

# SET TO NONE TO RUN ON FULL DATASET
NUM_EVAL_SAMPLES = None 

# ==========================================
# Step 1: Get the dataset
# ==========================================
print("--- Step 1: Loading Dataset ---")

# Download data if missing
if not os.path.exists("ori_pqal.json"):
    print("Downloading dataset...")
    os.system("wget https://raw.githubusercontent.com/pubmedqa/pubmedqa/refs/heads/master/data/ori_pqal.json")

# Load JSON
tmp_data = pd.read_json("ori_pqal.json").T

# Filter: keep only yes/no answers
tmp_data = tmp_data[tmp_data.final_decision.isin(["yes", "no"])]

# Prepare Documents (Contexts)
# Concatenating contexts and long_answer as abstract
documents = pd.DataFrame({
    "abstract": tmp_data.apply(lambda row: (" ").join(row.CONTEXTS + [row.LONG_ANSWER]), axis=1),
    "year": tmp_data.YEAR
})

# Prepare Questions
questions = pd.DataFrame({
    "question": tmp_data.QUESTION,
    "year": tmp_data.YEAR,
    "gold_label": tmp_data.final_decision,
    "gold_context": tmp_data.LONG_ANSWER,
    "gold_document_id": documents.index
})

print(f"Total documents: {len(documents)}")
print(f"Total questions: {len(questions)}")

# ==========================================
# Step 2: Configure LangChain LM
# ==========================================
print("\n--- Step 2: Configuring Language Model ---")

# Determine task type based on model architecture
task_type = "text2text-generation" if "t5" in MODEL_ID else "text-generation"

# Initialize HuggingFace Pipeline
pipe = pipeline(
    task_type,
    model=MODEL_ID,
    token=HF_TOKEN,
    device_map="auto",  # Automatically uses GPU
    max_new_tokens=20,  # Short output for Yes/No
    model_kwargs={"temperature": 0.01}  # Deterministic output
)

# Wrap in LangChain
# return_full_text=False ensures we get only the answer
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"return_full_text": False})

# Sanity Check
print("LM Check (Sky blue?):", llm.invoke("Is the sky usually blue? Answer Yes or No."))

# ==========================================
# Step 3: Set up the document database
# ==========================================
print("\n--- Step 3: Setting up Vector Store ---")

# 3.1 Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_ID, model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'})

# 3.2 Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Create documents with Metadata (Critical for evaluation)
metadatas = [{"id": idx} for idx in documents.index]
texts = text_splitter.create_documents(
    texts=documents.abstract.tolist(), 
    metadatas=metadatas
)
print(f"Created {len(texts)} chunks.")

# 3.3 Vector Store (Chroma)
print("Building Chroma Vector Store (Index)...")
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embedding_model,
    collection_name="pubmed_qa_full"
)
print("Vector Store Ready.")

# ==========================================
# Step 4: Define RAG Pipeline (Option B)
# ==========================================
print("\n--- Step 4: Defining RAG Pipeline (LCEL) ---")

# Retriever: fetch 1 document
retriever = vector_store.as_retriever(search_kwargs={"k": 1}) 

# Prompt Template
# Explicitly asking for Yes/No to make evaluation easier
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know.
Answer ONLY with "yes" or "no".

Context: {context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

# Generation Chain
generation_chain = (
    prompt
    | llm
    | output_parser
)

# RAG Chain (Parallel execution to keep context)
rag_chain = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=generation_chain)

# ==========================================
# Step 5: Evaluate RAG on the dataset
# ==========================================
print("\n--- Step 5: Full Evaluation ---")

# Determine evaluation subset
if NUM_EVAL_SAMPLES is None:
    eval_data = questions
    print(f"Running on FULL dataset ({len(eval_data)} samples)...")
else:
    eval_data = questions.iloc[:NUM_EVAL_SAMPLES]
    print(f"Running on SUBSET ({len(eval_data)} samples)...")

# Storage for results
rag_predictions = []
rag_valid_indices = []
retrieved_doc_ids = []
baseline_predictions = []

# Baseline Prompt (No Context)
baseline_prompt = ChatPromptTemplate.from_template(
    "Question: {question}\nAnswer ONLY with 'yes' or 'no'.\nAnswer:"
)
baseline_chain = baseline_prompt | llm | output_parser

start_time = time.time()

for idx, row in eval_data.iterrows():
    q = row['question']
    
    # --- A. RAG Inference ---
    try:
        res = rag_chain.invoke(q)
        ans = res['answer'].strip().lower()
        
        # Store Retrieved ID for Retrieval Accuracy
        retrieved_id = res['context'][0].metadata['id']
        retrieved_doc_ids.append(retrieved_id)
        
        # Standardize Answer
        if "yes" in ans:
            rag_predictions.append("yes")
            rag_valid_indices.append(idx)
        elif "no" in ans:
            rag_predictions.append("no")
            rag_valid_indices.append(idx)
        else:
            rag_predictions.append("invalid")
            
    except Exception as e:
        print(f"Error in RAG at index {idx}: {e}")
        rag_predictions.append("error")
        retrieved_doc_ids.append(None)

    # --- B. Baseline Inference ---
    try:
        base_res = baseline_chain.invoke({"question": q})
        base_ans = base_res.strip().lower()
        if "yes" in base_ans:
            baseline_predictions.append("yes")
        elif "no" in base_ans:
            baseline_predictions.append("no")
        else:
            baseline_predictions.append("invalid")
    except Exception as e:
        baseline_predictions.append("error")

    # Progress Log (every 50 samples)
    if (list(eval_data.index).index(idx) + 1) % 50 == 0:
        print(f"Processed {list(eval_data.index).index(idx) + 1}/{len(eval_data)} samples...")

total_time = time.time() - start_time
print(f"\nEvaluation completed in {total_time:.2f} seconds.")

# ==========================================
# Metrics Calculation
# ==========================================
print("\n=== FINAL RESULTS ===")

# 1. RAG Accuracy & F1
valid_golds = eval_data.loc[rag_valid_indices]['gold_label'].tolist()
valid_preds = [p for p in rag_predictions if p in ["yes", "no"]]

if len(valid_golds) > 0:
    rag_acc = accuracy_score(valid_golds, valid_preds)
    rag_f1 = f1_score(valid_golds, valid_preds, pos_label="yes", average="binary")
    print(f"RAG Accuracy: {rag_acc:.4f}")
    print(f"RAG F1 Score: {rag_f1:.4f}")
    print(f"Valid RAG Responses: {len(valid_golds)}/{len(eval_data)}")
else:
    print("No valid RAG responses.")

# 2. Baseline Accuracy
base_valid_mask = [p in ["yes", "no"] for p in baseline_predictions]
base_valid_preds = [p for p, m in zip(baseline_predictions, base_valid_mask) if m]
base_valid_golds = eval_data.loc[eval_data.index[base_valid_mask]]['gold_label'].tolist()

if len(base_valid_golds) > 0:
    base_acc = accuracy_score(base_valid_golds, base_valid_preds)
    print(f"Baseline Accuracy: {base_acc:.4f}")
    print(f"Valid Baseline Responses: {len(base_valid_golds)}/{len(eval_data)}")
else:
    print("No valid Baseline responses.")

# 3. Retrieval Accuracy
retrieval_hits = 0
for i, (idx, row) in enumerate(eval_data.iterrows()):
    # Compare retrieved ID with Gold ID
    if retrieved_doc_ids[i] == row['gold_document_id']:
        retrieval_hits += 1

retrieval_acc = retrieval_hits / len(eval_data)
print(f"Retrieval Accuracy: {retrieval_acc:.4f}")

# 4. Save detailed logs to file (Optional for report)
print("\nWriting predictions to 'evaluation_log.csv'...")
results_df = eval_data.copy()
results_df['rag_pred'] = rag_predictions
results_df['baseline_pred'] = baseline_predictions
results_df['retrieved_id'] = retrieved_doc_ids
results_df.to_csv("evaluation_log.csv", index=True)
print("Done.")