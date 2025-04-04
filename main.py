import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
from keras.preprocessing import image
from mtcnn import MTCNN
import skfuzzy as fuzz
import skfuzzy.control as ctrl

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


def integrate_cnn_fuzzy(cnn_model, image_path, fuzzy_system):
    # Extract the face parts from the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    faces = detector.detect_faces(image_rgb)

    if faces:
        keypoints = faces[0]['keypoints']
        parts = extract_face_parts(image_rgb, keypoints)

        # Process the face parts and prepare the data for CNN prediction
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

        # Predict with the CNN model
        cnn_predictions = cnn_model.predict(data)
        cnn_result = np.mean(cnn_predictions)  # Take the mean if you have multiple predictions

        # Use the CNN result as input to the fuzzy system for autism risk
        fuzzy_system.input['left_eye_intensity'] = np.mean(data[..., 0]) / 255
        fuzzy_system.input['right_eye_intensity'] = np.mean(data[..., 1]) / 255
        fuzzy_system.input['nose_intensity'] = np.mean(data[..., 2]) / 255
        fuzzy_system.input['lips_intensity'] = np.mean(data[..., 3]) / 255
        fuzzy_system.input['forehead_intensity'] = np.mean(data[..., 4]) / 255
        fuzzy_system.input['chin_intensity'] = np.mean(data[..., 5]) / 255
        fuzzy_system.input['cheek_intensity'] = np.mean(data[..., 6]) / 255

        fuzzy_system.compute()
        autism_risk_value = fuzzy_system.output['autism_risk']

        # Determine significant features based on autism risk
        significant_features = []
        if autism_risk_value > 50:
            significant_features = ['left_eye', 'right_eye', 'nose']  # Example, adjust based on logic

        return cnn_result, autism_risk_value
    else:
        return 0, 0  # No face detected

    
def create_fuzzy_system():
    # Define fuzzy sets for facial features
    features = [
        'left_eye_intensity', 'right_eye_intensity', 'nose_intensity',
        'lips_intensity', 'forehead_intensity', 'chin_intensity', 'cheek_intensity'
    ]
    
    # Define fuzzy inputs
    fuzzy_inputs = {feature: ctrl.Antecedent(np.arange(0, 1.01, 0.01), feature) for feature in features}
    
    # Define fuzzy output (autism risk)
    autism_risk = ctrl.Consequent(np.arange(0, 101, 1), 'autism_risk')

    # Define fuzzy sets for inputs
    for feature in fuzzy_inputs.values():
        feature['very_low'] = fuzz.gaussmf(feature.universe, 0, 0.1)
        feature['low'] = fuzz.gaussmf(feature.universe, 0.25, 0.15)
        feature['medium'] = fuzz.gaussmf(feature.universe, 0.5, 0.2)
        feature['high'] = fuzz.gaussmf(feature.universe, 0.75, 0.15)
        feature['very_high'] = fuzz.gaussmf(feature.universe, 1, 0.1)

    # Define fuzzy sets for output (autism risk)
    autism_risk['very_low'] = fuzz.trimf(autism_risk.universe, [0, 0, 20])
    autism_risk['low'] = fuzz.trimf(autism_risk.universe, [10, 30, 50])
    autism_risk['medium'] = fuzz.trimf(autism_risk.universe, [40, 60, 80])
    autism_risk['high'] = fuzz.trimf(autism_risk.universe, [70, 90, 100])
    autism_risk['very_high'] = fuzz.trimf(autism_risk.universe, [90, 100, 100])

    # Define fuzzy rules for autism risk
    rules = [
        ctrl.Rule(fuzzy_inputs['left_eye_intensity']['high'] & fuzzy_inputs['right_eye_intensity']['high'], autism_risk['high']),
        ctrl.Rule(fuzzy_inputs['lips_intensity']['low'] & fuzzy_inputs['chin_intensity']['low'], autism_risk['medium']),
        ctrl.Rule(fuzzy_inputs['forehead_intensity']['medium'] | fuzzy_inputs['cheek_intensity']['medium'], autism_risk['low']),
        ctrl.Rule(fuzzy_inputs['nose_intensity']['low'] & fuzzy_inputs['lips_intensity']['high'], autism_risk['medium']),
        ctrl.Rule(fuzzy_inputs['cheek_intensity']['high'], autism_risk['high']),
        ctrl.Rule(
            (fuzzy_inputs['left_eye_intensity']['medium'] & fuzzy_inputs['right_eye_intensity']['medium']) |
            (fuzzy_inputs['nose_intensity']['high'] & fuzzy_inputs['lips_intensity']['medium']),
            autism_risk['medium']
        ),
        ctrl.Rule(
            (fuzzy_inputs['forehead_intensity']['low'] & fuzzy_inputs['cheek_intensity']['low']) & fuzzy_inputs['chin_intensity']['low'],
            autism_risk['very_high']
        )
    ]

    # Control system for autism risk prediction
    autism_ctrl = ctrl.ControlSystem(rules)
    autism_sim = ctrl.ControlSystemSimulation(autism_ctrl)

    return autism_sim

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Save uploaded image
        filename = "uploaded_image.jpg"
        with open(filename, "wb") as buffer:
            buffer.write(await file.read())

        # Create fuzzy system
        fuzzy_system = create_fuzzy_system()

        # Get predictions from CNN + fuzzy
        cnn_result, autism_risk = integrate_cnn_fuzzy(face_model, filename, fuzzy_system)

        # Delete the uploaded file
        os.remove(filename)

        # Final combined result logic
        if cnn_result > 0.5 and autism_risk > 50:
            combined_result = 'Autistic (High Risk)'
        elif cnn_result > 0.5 and autism_risk <= 50:
            combined_result = 'Autistic (Low Risk)'
        elif cnn_result <= 0.5 and autism_risk > 50:
            combined_result = 'Non-Autistic (High Risk)'
        else:
            combined_result = 'Non-Autistic (Low Risk)'
        return {
            "autistic": combined_result
        }

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
