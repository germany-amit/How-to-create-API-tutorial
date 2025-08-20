import streamlit as st
import yaml
import io
import zipfile
import json
from datetime import datetime
from jinja2 import Template

# Optional LLM imports
try:
    from gpt4all import GPT4All
except ImportError:
    GPT4All = None

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

st.set_page_config(page_title="Agentic AI API Engineer", layout="wide")

# --- Core Generators -------------------------------------------------------

def generate_openapi(api_name, version, desc, endpoints):
    spec = {
        'openapi': '3.0.3',
        'info': {'title': api_name, 'version': version, 'description': desc},
        'servers': [{'url': '{{baseUrl}}'}],
        'paths': {}
    }
    for ep in endpoints:
        path = ep['path']
        method = ep['method'].lower()
        spec['paths'].setdefault(path, {})[method] = {
            'summary': ep.get('summary', ''),
            'responses': {'200': {'description': 'Success'}}
        }
    return yaml.safe_dump(spec, sort_keys=False)

def scaffold_fastapi_app(api_name, version, endpoints):
    template = Template("""
from fastapi import FastAPI

app = FastAPI(title="{{title}}", version="{{version}}")

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": "{{timestamp}}"}

{% for ep in endpoints %}
@app.{{ep.method.lower()}}("{{ep.path}}")
async def {{ep.func_name}}():
    return {"demo": "{{ep.summary}}"}
{% endfor %}
""")
    return template.render(title=api_name, version=version, endpoints=endpoints, timestamp=datetime.utcnow().isoformat())

def scaffold_client_demo(endpoints):
    code = ["""import requests
BASE_URL = "http://localhost:8000"
"""]
    for ep in endpoints:
        code.append(f"resp = requests.{ep['method'].lower()}(f'{{BASE_URL}}{ep['path']}')")
        code.append("print(resp.status_code, resp.text)")
    return "\n".join(code)

def make_zip(files: dict):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, content in files.items():
            z.writestr(fname, content)
    buf.seek(0)
    return buf

# --- Fallback / Rule-based Generator --------------------------------------

def rule_based_generate(user_prompt):
    """
    Basic fallback API endpoints
    """
    return [
        {"path": "/items", "method": "GET", "summary": "List items", "func_name": "list_items"},
        {"path": "/items", "method": "POST", "summary": "Create item", "func_name": "create_item"},
    ]

# --- Lightweight LLM Functions --------------------------------------------

def gpt4all_generate(prompt):
    if GPT4All is None:
        return rule_based_generate(prompt)
    try:
        model = GPT4All("gpt4all-lora-quantized.bin")  # small CPU-friendly model
        response = model.generate(prompt)
        return json.loads(response)  # expect JSON output
    except Exception:
        return rule_based_generate(prompt)

def llama_cpp_generate(prompt):
    if Llama is None:
        return rule_based_generate(prompt)
    try:
        model = Llama(model_path="llama-7b-q4.bin")  # small local LLaMA
        response = model(prompt, max_tokens=200)
        return json.loads(response['choices'][0]['text'])
    except Exception:
        return rule_based_generate(prompt)

# --- Prebuilt Demos --------------------------------------------------------

demo_apis = {
    "Todo API": [
        {"path": "/todos", "method": "GET", "summary": "List todos", "func_name": "list_todos"},
        {"path": "/todos", "method": "POST", "summary": "Create todo", "func_name": "create_todo"},
    ],
    "Notes API": [
        {"path": "/notes", "method": "GET", "summary": "List notes", "func_name": "list_notes"},
        {"path": "/notes", "method": "POST", "summary": "Create note", "func_name": "create_note"},
    ],
    "Calculator API": [
        {"path": "/add", "method": "GET", "summary": "Add two numbers", "func_name": "add"},
        {"path": "/multiply", "method": "GET", "summary": "Multiply two numbers", "func_name": "multiply"},
    ]
}

# --- UI --------------------------------------------------------------------

st.title("Agentic AI API Engineer — Free LLM Parallel + Fallback")
st.caption("Generate simple APIs live using lightweight open-source LLMs or fallback rules.")

if 'custom_req' not in st.session_state:
    st.session_state.custom_req = "I want a simple API that manages tasks."

choice = st.selectbox("Choose Demo API", list(demo_apis.keys()) + ["Custom requirement"])

if choice != "Custom requirement":
    endpoints = demo_apis[choice]
    api_name = choice
    version = "0.1.0"
    desc = f"Auto-generated {choice} using free LLM engine."
    st.session_state.custom_req = "I want a simple API that manages tasks."
else:
    api_name = st.text_input("API Name", value="Custom API", disabled=True)
    version = st.text_input("Version", value="0.1.0", disabled=True)
    desc = st.text_area("Description", value="Generated from natural language requirement.", disabled=True)
    st.session_state.custom_req = st.text_area("Enter requirement", value=st.session_state.custom_req, height=150)
    endpoints = []

if st.button("Generate API from scratch"):
    if choice == "Custom requirement":
        if not st.session_state.custom_req:
            st.error("Please enter a requirement to generate a custom API.")
            st.stop()

        with st.spinner("Generating API endpoints using LLMs..."):
            # Run both lightweight LLMs in parallel (simulated)
            gpt4all_endpoints = gpt4all_generate(st.session_state.custom_req)
            llama_endpoints = llama_cpp_generate(st.session_state.custom_req)

        st.subheader("GPT4All Generated Endpoints")
        st.json(gpt4all_endpoints)

        st.subheader("LLaMA-Cpp Generated Endpoints")
        st.json(llama_endpoints)

        # Let user pick one (for simplicity, pick GPT4All if exists)
        endpoints = gpt4all_endpoints if gpt4all_endpoints else llama_endpoints
        if not endpoints:
            endpoints = rule_based_generate(st.session_state.custom_req)

    # Generate outputs
    openapi_yaml = generate_openapi(api_name, version, desc, endpoints)
    fastapi_code = scaffold_fastapi_app(api_name, version, endpoints)
    client_code = scaffold_client_demo(endpoints)

    st.subheader("OpenAPI Spec")
    st.code(openapi_yaml, language='yaml')

    st.subheader("FastAPI Code")
    st.code(fastapi_code, language='python')

    st.subheader("Client Demo (how to use the API)")
    st.code(client_code, language='python')

    st.subheader("Try API Now (simulated)")
    for ep in endpoints:
        st.button(f"Call {ep['method']} {ep['path']}")
        st.json({"message": f"Called {ep['method']} {ep['path']}", "summary": ep['summary'], "status": "ok"})

    files = {
        'openapi.yaml': openapi_yaml,
        'backend/main.py': fastapi_code,
        'client_demo.py': client_code,
        'README.md': f"# {api_name}\n\n{desc}\n"
    }
    zipbuf = make_zip(files)
    st.download_button("Download API Project ZIP", zipbuf, file_name=f"{api_name.replace(' ','_')}_cradle.zip")

st.info("Cradle-to-cradle API engineering: requirement → spec → backend → client demo → simulated try.")
