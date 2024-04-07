from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List
from datetime import datetime
from uuid import uuid4

app = FastAPI()

# Mock database
users_db = {}
referrals_db = {}

class User(BaseModel):
    name: str
    email: str
    password: str
    referral_code: str = None

class UserDetails(BaseModel):
    name: str
    email: str
    referral_code: str
    timestamp: datetime

class Referral(BaseModel):
    name: str
    email: str
    timestamp: datetime

def get_auth_token(token: str = Header(...)):
    # Dummy authentication logic
    if token != "valid_token":
        raise HTTPException(status_code=401, detail="Unauthorized")

@app.post("/register/", response_model=UserDetails)
async def register_user(user: User):
    user_id = str(uuid4())
    timestamp = datetime.now()
    user_data = {
        "name": user.name,
        "email": user.email,
        "password": user.password,
        "referral_code": user.referral_code,
        "timestamp": timestamp
    }
    users_db[user_id] = user_data
    
    # If there's a referral code, add it to the referrals_db
    if user.referral_code:
        referrals_db.setdefault(user.referral_code, []).append({
            "name": user.name,
            "email": user.email,
            "timestamp": timestamp
        })
    
    return user_data

@app.get("/details/", response_model=UserDetails)
async def get_user_details(auth=Depends(get_auth_token)):
    # Dummy logic to retrieve user details from auth token
    user_id = "dummy_user_id"
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return users_db[user_id]

@app.get("/referrals/", response_model=List[Referral])
async def get_user_referrals(auth=Depends(get_auth_token)):
    # Dummy logic to retrieve user details from auth token
    user_id = "dummy_user_id"
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    referral_code = users_db[user_id]["referral_code"]
    if not referral_code:
        raise HTTPException(status_code=404, detail="User has no referrals")
    if referral_code not in referrals_db:
        return []
    return referrals_db[referral_code]

