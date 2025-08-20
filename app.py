import streamlit as st
import yaml
import io
import zipfile
import requests
import json
import os
import google.generativeai as genai
from datetime import datetime
from jinja2 import Template

st.set_page_config(page_title="Agentic AI API Engineer", layout="wide")

# Configure the Gemini API key from Streamlit secrets
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    # Use gemini-2.5-flash-preview-05-20 as the model for text generation
    MODEL = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
except KeyError:
    st.error("API Key not found. Please add your GEMINI_API_KEY to Streamlit's secrets manager.")

# --- Core Generators (Unchanged) --------------------------------------------

def generate_openapi(api_name, version, desc, endpoints):
    """
    Generates an OpenAPI 3.0.3 specification in YAML format.
    
    Args:
        api_name (str): The name of the API.
        version (str): The version of the API.
        desc (str): A description of the API.
        endpoints (list): A list of dictionaries, each representing an endpoint.
    
    Returns:
        str: The OpenAPI specification in YAML format.
    """
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
    """
    Scaffolds a basic FastAPI application from a Jinja2 template.
    
    Args:
        api_name (str): The name of the API.
        version (str): The version of the API.
        endpoints (list): A list of dictionaries, each representing an endpoint.
    
    Returns:
        str: The Python code for the FastAPI application.
    """
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
    """
    Scaffolds a basic Python client demo using the 'requests' library.
    
    Args:
        endpoints (list): A list of dictionaries, each representing an endpoint.
    
    Returns:
        str: The Python code for the client demo.
    """
    code = ["""import requests
BASE_URL = "http://localhost:8000"
"""]
    for ep in endpoints:
        code.append(f"resp = requests.{ep['method'].lower()}(f'{{BASE_URL}}{ep['path']}')")
        code.append("print(resp.status_code, resp.text)")
    return "\n".join(code)

def make_zip(files: dict):
    """
    Creates a in-memory zip file from a dictionary of file names and content.
    
    Args:
        files (dict): A dictionary where keys are file names and values are file content.
    
    Returns:
        io.BytesIO: An in-memory buffer containing the zip file.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, content in files.items():
            z.writestr(fname, content)
    buf.seek(0)
    return buf

# --- AI-Powered Core Functionality (New) -----------------------------------

def generate_api_details_from_prompt(user_prompt):
    """
    Uses the Gemini API to generate API details (name, version, endpoints) 
    from a natural language prompt. It asks the model for a JSON response.
    
    Args:
        user_prompt (str): The natural language description of the API.
    
    Returns:
        dict: A dictionary containing the generated API details.
    """
    prompt = f"""
    You are a professional API designer. Based on the following user request, provide a JSON object
    containing the API details. The JSON object must have a 'name' (string), 'version' (string),
    'description' (string), and an 'endpoints' (array of objects). Each endpoint object must have
    'path' (string), 'method' (string, e.g., 'GET' or 'POST'), 'summary' (string),
    and 'func_name' (a valid Python function name, lowercase_with_underscores).
    
    The response must be a valid JSON object and nothing else.
    
    User request: {user_prompt}
    """
    
    # We use a generation config to force the model to return a JSON object
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string"},
                "description": {"type": "string"},
                "endpoints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "method": {"type": "string"},
                            "summary": {"type": "string"},
                            "func_name": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["name", "version", "description", "endpoints"]
        }
    }

    response = MODEL.generate_content(prompt, generation_config=generation_config)
    
    return json.loads(response.text)

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

st.title("Agentic AI API Engineer — Cradle to Cradle")
st.caption("Generate, run, and try simple APIs live from requirements.")

# Using session state to manage the custom requirement text area
if 'custom_req' not in st.session_state:
    st.session_state.custom_req = "I want a simple API that manages tasks."

choice = st.selectbox("Choose Demo API", list(demo_apis.keys()) + ["Custom requirement"])

if choice != "Custom requirement":
    endpoints = demo_apis[choice]
    api_name = choice
    version = "0.1.0"
    desc = f"Auto-generated {choice} using Agentic AI API Engineer."
    st.session_state.custom_req = "I want a simple API that manages tasks." # Reset the text area
else:
    api_name = st.text_input("API Name", value="Custom API", disabled=True)
    version = st.text_input("Version", value="0.1.0", disabled=True)
    desc = st.text_area("Description", value="Generated from natural language requirement.", disabled=True)
    st.session_state.custom_req = st.text_area("Enter requirement", value=st.session_state.custom_req, height=150)
    endpoints = [] # Will be generated by AI later

if st.button("Generate API from scratch"):
    if choice == "Custom requirement":
        if not st.session_state.custom_req:
            st.error("Please enter a requirement to generate a custom API.")
            st.stop()
        
        with st.spinner("Generating API details with AI..."):
            try:
                # Call the new AI function to get the API details
                api_details = generate_api_details_from_prompt(st.session_state.custom_req)
                
                # Extract the details from the AI response
                api_name = api_details.get("name", "Custom API")
                version = api_details.get("version", "0.1.0")
                desc = api_details.get("description", "Generated by AI.")
                endpoints = api_details.get("endpoints", [])
                
                if not endpoints:
                    st.warning("AI did not generate any endpoints. Please try a different prompt.")
                    st.stop()
                
            except Exception as e:
                st.error(f"An error occurred during AI generation: {e}")
                st.stop()

    if endpoints: # Ensure endpoints exist before proceeding
        openapi_yaml = generate_openapi(api_name, version, desc, endpoints)
        fastapi_code = scaffold_fastapi_app(api_name, version, endpoints)
        client_code = scaffold_client_demo(endpoints)

        st.subheader("OpenAPI Spec")
        st.code(openapi_yaml, language='yaml')

        st.subheader("FastAPI Code")
        st.code(fastapi_code, language='python')

        st.subheader("Client Demo (how to use the API)")
        st.code(client_code, language='python')

        st.subheader("Try API Now")
        # In a real-world scenario, you would deploy the API to a live server to try it.
        # This is a client-side mock to simulate the result.
        for ep in endpoints:
            st.button(f"Call {ep['method']} {ep['path']}")
            # The next line is a mock response, a live API would require a deployed service.
            st.json({"message": f"Called {ep['method']} {ep['path']}", "summary": ep['summary'], "status": "ok"})

        files = {
            'openapi.yaml': openapi_yaml,
            'backend/main.py': fastapi_code,
            'client_demo.py': client_code,
            'README.md': f"# {api_name}\n\n{desc}\n"
        }
        zipbuf = make_zip(files)
        st.download_button("Download API Project ZIP", zipbuf, file_name=f"{api_name.replace(' ','_')}_cradle.zip")
    else:
        st.warning("Please select a demo or enter a custom requirement to generate the API.")

