import secrets
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from Reword_prompt.reword import reword

# Import the get_tokens function from your database helper
from generate_auth_tokens.models.create_database import init_db, save_token, get_tokens, delete_token

from rag import rag_request

from dotenv import load_dotenv
import os
load_dotenv()  # Load .env variables
groq_api_key = os.getenv("GROQ_API_KEY")

app = FastAPI()

# Mount static files (CSS, etc.)
app.mount(
    "/static",
    StaticFiles(directory="generate_auth_tokens/static"),
    name="static"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set the templates directory
templates = Jinja2Templates(directory="generate_auth_tokens/templates")

# Initialize database when the app starts
init_db()

# Define the request schema
class InputPrompt(BaseModel):
    input_prompt: str

def is_valid_token(token: str) -> bool:
    """Check if the provided token exists in the database."""
    tokens = get_tokens()  # returns a list of (id, token, created_at)
    valid_tokens = [t[1] for t in tokens]  # extract token strings
    return token in valid_tokens

def get_current_token(access_token: str = Header(...)):
    """Dependency that validates the provided access token."""
    if not is_valid_token(access_token):
        raise HTTPException(status_code=401, detail="Invalid auth token")
    return access_token

@app.post("/demo")
async def demo(data: InputPrompt, token: str = Depends(get_current_token)):
    # Extract input data
    input_prompt = data.input_prompt

    try:
        # Call the function and generate the output
        print("Input prompt:", input_prompt)
        output_reworded_prompt = reword(input_prompt)
        print("Output reworded prompt:", output_reworded_prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"Reworded_prompt": output_reworded_prompt}

@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    """Render the main page with the token generation form."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
async def generate_token(request: Request):
    """Generate a token with a fixed prefix, save it, and render the result page."""
    token = "incept-" + secrets.token_urlsafe(16)
    save_token(token)
    return templates.TemplateResponse("result.html", {"request": request, "token": token})

@app.get("/tokens", response_class=HTMLResponse)
async def display_tokens(request: Request):
    """Display all tokens with options to remove them."""
    tokens = get_tokens()
    return templates.TemplateResponse("tokens.html", {"request": request, "tokens": tokens})

@app.post("/delete_token", response_class=HTMLResponse)
async def remove_token(request: Request):
    """Remove a token based on its ID and then display the updated token list."""
    form_data = await request.form()
    token_id = int(form_data.get("token_id"))
    delete_token(token_id)
    tokens = get_tokens()
    return templates.TemplateResponse("tokens.html", {"request": request, "tokens": tokens})

@app.post("/rag")
async def rag(data: InputPrompt, token: str = Depends(get_current_token)):
    # Extract input data
    input_prompt = data.input_prompt

    try:
        # Call the function and generate the output
        print("Input prompt:", input_prompt)
        output_rag_response = rag_request(groq_api_key, input_prompt,
                                          tfidf_top_n=5,
                                          chroma_top_n=5,
                                          final_top_k=3,
                                          persist_dir=r"C:\profolders\Internships\Inceptai\rag\RA\chromadb_data",
                                          collection_name="kql_context_embeddings")
        print("Output response from RAG:", output_rag_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"RAG_response": output_rag_response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
