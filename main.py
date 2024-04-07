from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List
from datetime import datetime
from uuid import uuid4

app = FastAPI()

# Mock database
users_db = {}
referrals_db = {}
auth_tokens = {}  # Store authentication tokens

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
    token: str  # Include token in UserDetails response

class Referral(BaseModel):
    name: str
    email: str
    timestamp: datetime

def generate_token():
    return str(uuid4())

@app.post("/register/", response_model=UserDetails)
async def register_user(user: User):
    user_id = str(uuid4())
    timestamp = datetime.now()
    token = generate_token()  # Generate token for the user
    user_data = {
        "name": user.name,
        "email": user.email,
        "password": user.password,
        "referral_code": user.referral_code,
        "timestamp": timestamp,
        "token": token
    }
    users_db[user_id] = user_data
    auth_tokens[token] = user_id  # Store token with user_id
    
    # If there's a referral code, add it to the referrals_db
    if user.referral_code:
        referrals_db.setdefault(user.referral_code, []).append({
            "name": user.name,
            "email": user.email,
            "timestamp": timestamp
        })
    
    return user_data

def get_user_id_from_token(token: str = Header(...)):
    if token not in auth_tokens:
        raise HTTPException(status_code=401, detail="Invalid token")
    return auth_tokens[token]

@app.get("/details/", response_model=UserDetails)
async def get_user_details(user_id: str = Depends(get_user_id_from_token)):
    # Retrieve user details using user_id
    user_data = users_db.get(user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    return user_data

@app.get("/referrals/", response_model=List[Referral])
async def get_user_referrals(user_id: str = Depends(get_user_id_from_token)):
    # Dummy logic to retrieve user details from auth token
    user_data = users_db.get(user_id)
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    referral_code = user_data["referral_code"]
    if not referral_code:
        raise HTTPException(status_code=404, detail="User has no referrals")
    if referral_code not in referrals_db:
        return []
    return referrals_db[referral_code]
