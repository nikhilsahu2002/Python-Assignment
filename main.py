from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import JSONResponse
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from bson.objectid import ObjectId
from pydantic import BaseModel


app = FastAPI()

# Load the trained model
model_path = "Autisum_Detector_Model_main_Epoch_50.h5"
model = load_model(model_path)

# MongoDB connection URL
MONGODB_URL = "mongodb+srv://death1233freak:OY10LK1hpxaSOmUs@cluster0.4uc8s20.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB database
client = AsyncIOMotorClient(MONGODB_URL)
db = client["autism_detection_db"]

# Define MongoDB model using Pydantic
class PredictionResult(BaseModel):
    result: str

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
        
        # Determine result
        result = "The MRI image has Autism." if prediction[0, 0] >= 0.5 else "The MRI image does not have Autism."
        
        # Save image data to MongoDB as binary
        with open(temp_file_path, "rb") as img_file:
            img_data = img_file.read()
            image_id = await db.images.insert_one({"data": img_data})
        
        # Save prediction result to MongoDB along with image ID
        predictions_collection = db["predictions"]
        await predictions_collection.insert_one({"result": result, "image_id": str(image_id.inserted_id)})
        
        # Delete the temporary file
        os.remove(temp_file_path)
        
        return PredictionResult(result=result)
    
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to retrieve and view image by ID
@app.get("/view_image/{image_id}")
async def view_image(image_id: str):
    try:
        # Retrieve image data from MongoDB
        image_data = await db.images.find_one({"_id": ObjectId(image_id)})
        if image_data:
            # Convert image data back to bytes
            img_bytes = image_data["data"]
            # Return the image
            return Response(content=img_bytes, media_type="image/jpeg")
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
