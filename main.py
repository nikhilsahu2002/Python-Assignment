import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from keras.preprocessing import image
from mtcnn import MTCNN

app = FastAPI()

#none
# Load the trained models
face_model_path = "Face_Model.h5"
mri_model_path = "Mri_Model.h5"
face_model = load_model(face_model_path)
mri_model = load_model(mri_model_path)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "https://autisum.vercel.app/", "*"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

detector = MTCNN()

def extract_face_parts(image, keypoints):
    parts = {}
    for part, (x, y) in keypoints.items():
        if part == 'left_eye':
            parts['left_eye'] = image[y-30:y+30, x-30:x+30]
        elif part == 'right_eye':
            parts['right_eye'] = image[y-30:y+30, x-30:x+30]
        elif part == 'nose':
            parts['nose'] = image[y-30:y+30, x-30:x+30]
        elif part == 'mouth_left' or part == 'mouth_right':
            parts['lips'] = image[y-20:y+20, x-40:x+40]

    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']
    nose = keypoints['nose']
    mouth_left = keypoints['mouth_left']
    mouth_right = keypoints['mouth_right']
    
    upper_y = min(left_eye[1], right_eye[1]) - 50
    upper_x1 = min(left_eye[0], right_eye[0]) - 50
    upper_x2 = max(left_eye[0], right_eye[0]) + 50
    parts['upper_face'] = image[upper_y:min(left_eye[1], right_eye[1])+30, upper_x1:upper_x2]

    middle_y = nose[1] - 40
    middle_x1 = nose[0] - 50
    middle_x2 = nose[0] + 50
    parts['middle_face'] = image[middle_y:nose[1]+40, middle_x1:middle_x2]

    philtrum_y = nose[1]
    philtrum_x1 = mouth_left[0]
    philtrum_x2 = mouth_right[0]
    parts['philtrum'] = image[philtrum_y-20:philtrum_y+20, philtrum_x1:philtrum_x2]
    
    return parts

def is_autistic_face(image_path, model):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)

    if faces:
        keypoints = faces[0]['keypoints']
        parts = extract_face_parts(image_rgb, keypoints)
        num_samples = min(len(parts['left_eye']), len(parts['right_eye']), len(parts['nose']), 
                          len(parts['lips']), len(parts['upper_face']), len(parts['middle_face']), 
                          len(parts['philtrum']))

        data = np.zeros((num_samples, 64, 64, 7), dtype=np.uint8)
        data[..., 0] = np.array([cv2.resize(img, (64, 64)) for img in parts['left_eye'][:num_samples]])
        data[..., 1] = np.array([cv2.resize(img, (64, 64)) for img in parts['right_eye'][:num_samples]])
        data[..., 2] = np.array([cv2.resize(img, (64, 64)) for img in parts['nose'][:num_samples]])
        data[..., 3] = np.array([cv2.resize(img, (64, 64)) for img in parts['lips'][:num_samples]])
        data[..., 4] = np.array([cv2.resize(img, (64, 64)) for img in parts['upper_face'][:num_samples]])
        data[..., 5] = np.array([cv2.resize(img, (64, 64)) for img in parts['middle_face'][:num_samples]])
        data[..., 6] = np.array([cv2.resize(img, (64, 64)) for img in parts['philtrum'][:num_samples]])

        prediction = model.predict(data)
        return bool(np.mean(prediction) >= 0.5)
    else:
        return False

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        filename = "uploaded_image.jpg"
        with open(filename, "wb") as buffer:
            buffer.write(await file.read())

        result = is_autistic_face(filename, face_model)
        os.remove(filename)

        return {"autistic": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    try:
        temp_file_path = 'temp_img.jpg'
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(await file.read())
        
        img = image.load_img(temp_file_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = mri_model.predict(img_array)
        result = "The MRI image has Autism." if prediction[0, 0] >= 0.5 else "The MRI image does not have Autism."
        os.remove(temp_file_path)

        return JSONResponse({"result": result})
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
