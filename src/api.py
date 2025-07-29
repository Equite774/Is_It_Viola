import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from CNN_final.model import load_model, make_prediction
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
from api_utils import preprocess

app = FastAPI()
model = load_model()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Validate the file
    extension = file.filename.lower().split('.')[-1]
    allowed_extensions = {"wav", "mp3", "m4a"}
    if not file or extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    
    # Preprocess the audio file
    contents = await file.read()
    audio = BytesIO(contents)
    list_of_spectrograms = preprocess(audio)
    
    # Make predictions
    probabilities = make_prediction(model, list_of_spectrograms)

    # Return the probabilities
    return JSONResponse(content={"probabilities": probabilities})