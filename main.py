from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from Reword_prompt.reword import reword

# Define the FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the request schema
class InputPrompt(BaseModel):
    input_prompt : str


@app.post("/demo")
async def demo(data: InputPrompt):    
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)