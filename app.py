import gradio as gr
from llama_cpp import Llama
import os
import threading
import queue
import subprocess
import sys

from langchain_community.document_loaders import GitLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

active_process = None
log_queue = queue.Queue()
log_history = ""
db = None 

LANG_MAP = {
    "python": {"ext": ".py", "cmd": ["python"]},
    "c": {"ext": ".c", "compile": ["gcc", "sandbox.c", "-o", "sandbox"], "run": ["sandbox.exe" if os.name == 'nt' else "./sandbox"]},
    "cpp": {"ext": ".cpp", "compile": ["g++", "sandbox.cpp", "-o", "sandbox"], "run": ["sandbox.exe" if os.name == 'nt' else "./sandbox"]},
    "javascript": {"ext": ".js", "cmd": ["node"]},
    "rust": {"ext": ".rs", "compile": ["rustc", "sandbox.rs"], "run": ["sandbox.exe" if os.name == 'nt' else "./sandbox"]}
}

def get_base_path():
    if getattr(sys, 'frozen', False): return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_path()
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(WORKSPACE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def load_xecode_engine():
    model_path = os.path.join(MODELS_DIR, "DCoder_q4_k_m.gguf")
    if not os.path.exists(model_path): return None
    return Llama(model_path=model_path, n_ctx=2048, verbose=False)

llm = load_xecode_engine()

def initialize_rag(repo_url):
    global db
    repo_path = os.path.join(WORKSPACE_DIR, "github_repo")
    loader = GitLoader(repo_path=repo_path, clone_url=repo_url, branch="main")
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
    return "Repository indexed!"

def retrieve_context(query):
    if not db: return ""
    results = db.similarity_search(query, k=2)
    return "\n".join([doc.page_content for doc in results])

def stream_logs(proc):
    for line in iter(proc.stdout.readline, ''):
        log_queue.put(line)

def stop_execution():
    global active_process
    if active_process:
        active_process.terminate()
        active_process = None

def get_latest_logs():
    global log_history
    while not log_queue.empty():
        log_history += log_queue.get()
    return log_history

def execute_universal_code(lang_key, code):
    global active_process, log_history
    stop_execution()
    log_history = ""
    config = LANG_MAP.get(lang_key.lower())
    filename = os.path.join(WORKSPACE_DIR, f"sandbox{config['ext']}")
    with open(filename, "w", encoding="utf-8") as f: f.write(code)
    if "compile" in config:
        subprocess.run(config["compile"], cwd=WORKSPACE_DIR)
        run_cmd = config["run"]
    else:
        run_cmd = config["cmd"] + [filename]
    active_process = subprocess.Popen(run_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=WORKSPACE_DIR)
    threading.Thread(target=stream_logs, args=(active_process,), daemon=True).start()
    return "Executing..."

def viber_generate(lang, prompt):
    if llm is None: return "Model missing."
    context = retrieve_context(prompt)
    enhanced_prompt = f"Reference Code:\n{context}\n\nLanguage: {lang}\nTask: {prompt}\nCode:"
    response = llm(enhanced_prompt, max_tokens=2048, stop=["User:"])
    return response["choices"][0]["text"]

with gr.Blocks(css="footer {display: none !important;} :root { background-color: #050505; color: white; font-family: monospace; }") as xecode_app:
    gr.HTML("<h1>📟 XECODE: Universal RAG IDE</h1>")
    with gr.Tabs():
        with gr.Tab("AI Engine"):
            lang_selector = gr.Dropdown(choices=list(LANG_MAP.keys()), label="Language", value="python")
            inp = gr.Textbox(label="PROMPT")
            btn = gr.Button("GENERATE CODE")
            ai_out = gr.Code(label="AI Output")
            btn.click(viber_generate, inputs=[lang_selector, inp], outputs=ai_out)
        with gr.Tab("RAG Config"):
            repo_url = gr.Textbox(label="GitHub Repo URL")
            index_btn = gr.Button("Index Repository")
            status = gr.Textbox(label="Index Status")
            index_btn.click(initialize_rag, inputs=repo_url, outputs=status)
        with gr.Tab("IDE Workspace"):
            lang_run = gr.Dropdown(choices=list(LANG_MAP.keys()), label="Language", value="python")
            code_editor = gr.Code(language="python")
            compile_btn = gr.Button("RUN CODE")
            terminal = gr.Textbox(label="Terminal Output", interactive=False, lines=10)
            gr.Timer(0.5).tick(fn=get_latest_logs, outputs=terminal)
            lang_run.change(lambda x: gr.Code(language=x), inputs=lang_run, outputs=code_editor)
            compile_btn.click(execute_universal_code, inputs=[lang_run, code_editor], outputs=terminal)
            gr.Button("STOP").click(stop_execution)

if __name__ == "__main__":
    xecode_app.queue().launch(inbrowser=True)