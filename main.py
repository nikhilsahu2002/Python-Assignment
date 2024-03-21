from fastapi import FastAPI, HTTPException
from typing import List, Optional
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Define data model for water quality observation record
class WaterQualityObservation(BaseModel):
    location: dict
    date_time: str
    description: str
    parameters: dict

# Mock data storage (replace with database integration later)
observations = []

# Function to initialize dummy data
def initialize_dummy_data():
    global observations
    dummy_data = [
        {
            "location": {"latitude": 40.712776, "longitude": -74.005974},
            "date_time": "2024-03-19T15:00:00Z",
            "description": "early afternoon water quality observation at Nehru Park",
            "parameters": {"pH": 7.4, "conductivity": 250, "DO": 67, "contaminants": ["Lead", "Arsenic"]}
        },
        {
            "location": {"latitude": 42.3601, "longitude": -71.0589},
            "date_time": "2024-03-20T10:00:00Z",
            "description": "morning water quality observation at Boston Harbor",
            "parameters": {"pH": 7.2, "conductivity": 270, "DO": 70, "contaminants": ["Mercury", "Cadmium"]}
        }
    ]
    observations = [WaterQualityObservation(**data) for data in dummy_data]

# Initialize dummy data
initialize_dummy_data()

# Create operation
@app.post("/observations/")
async def create_observation(observation: WaterQualityObservation):
    observations.append(observation)
    return observation

# Read operation
@app.get("/observations/", response_model=List[WaterQualityObservation])
async def read_observations():
    return observations

@app.get("/observations/{observation_id}", response_model=WaterQualityObservation)
async def read_observation(observation_id: int):
    try:
        return observations[observation_id]
    except IndexError:
        raise HTTPException(status_code=404, detail="Observation not found")

# Update operation
@app.put("/observations/{observation_id}", response_model=WaterQualityObservation)
async def update_observation(observation_id: int, observation: WaterQualityObservation):
    try:
        observations[observation_id] = observation
        return observation
    except IndexError:
        raise HTTPException(status_code=404, detail="Observation not found")

# Delete operation
@app.delete("/observations/{observation_id}")
async def delete_observation(observation_id: int):
    try:
        del observations[observation_id]
        return {"message": "Observation deleted successfully"}
    except IndexError:
        raise HTTPException(status_code=404, detail="Observation not found")

# Search functionality
@app.get("/observations/search/")
async def search_observations(latitude: float, longitude: float, start_date: Optional[str] = None, end_date: Optional[str] = None):
    filtered_observations = [obs for obs in observations if obs.location["latitude"] == latitude and obs.location["longitude"] == longitude]
    return filtered_observations
