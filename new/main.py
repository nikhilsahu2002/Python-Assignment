from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# MongoDB connection settings
MONGODB_URL = "mongodb+srv://death1233freak:OY10LK1hpxaSOmUs@cluster0.4uc8s20.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
client = AsyncIOMotorClient(MONGODB_URL)
db = client["autism_detection_db"]

# Define MongoDB model using Pydantic
class PredictionResult(BaseModel):
    result: str

# Load the trained model
model_path = "Autisum_Detector_Model_main_Epoch_50.h5"
model = load_model(model_path)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# Function to process the uploaded image
@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    try:
        # Create a temporary file to save the uploaded image
        temp_file_path = 'temp_img.jpg'
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(await file.read())
        
        # Read the image file
        img = image.load_img(temp_file_path, target_size=(128, 128))
        
        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch size dimension
        img_array = img_array / 255.0  # Normalize pixel values
        
        # Make prediction
        prediction = model.predict(img_array)
        
        # Determine the result based on prediction
        result = "The MRI image has Autism." if prediction[0, 0] >= 0.5 else "The MRI image does not have Autism."
        
        
        # Save the image data to MongoDB as binary
        with open(temp_file_path, "rb") as img_file:
            img_data = img_file.read()
            image_id = await db.images.insert_one({"data": img_data})
        
        # Save prediction result to MongoDB along with image ID
        predictions_collection = db["predictions"]
        await predictions_collection.insert_one({"result": result, "image_id": str(image_id.inserted_id)})
        
        # Delete the temporary file
        os.remove(temp_file_path)
        
        # Return the prediction result to the frontend
        return JSONResponse({"result": result})
    
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
