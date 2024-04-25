from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load the trained model
model_path = "Autisum_Detector_Model_main_Epoch_50.h5"
model = load_model(model_path)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173","https://autisum.vercel.app/","*"],
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
        
        # Return the result
        result = "The MRI image has Autism." if prediction[0, 0] >= 0.5 else "The MRI image does not have Autism."
        
        # Delete the temporary file
        os.remove(temp_file_path)
        
        return JSONResponse({"result": result})
    
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
