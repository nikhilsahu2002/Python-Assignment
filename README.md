
# FastAPI Referral System API

This repository contains the implementation of a referral system API using FastAPI.

## API Endpoints

### 1. User Registration Endpoint

- **Method**: `POST`
- **URL**: `/register/`
- **Description**: Allows users to register with the system.
- **Request Body**:
  ```json
  {
      "name": "string",
      "email": "string",
      "password": "string",
      "referral_code": "string" (optional)
  }
  ```
- **Response**: 
  - Status Code: 200 OK
  - Body: Returns user details along with a generated authentication token.

### 2. User Details Endpoint

- **Method**: `GET`
- **URL**: `/details/`
- **Description**: Retrieves details of the authenticated user.
- **Headers**:
  - `Authorization`: Authentication token obtained during registration.
- **Response**:
  - Status Code: 200 OK
  - Body: Returns user details including name, email, referral code, and registration timestamp.

### 3. Referrals Endpoint

- **Method**: `GET`
- **URL**: `/referrals/`
- **Description**: Retrieves a list of referrals for the authenticated user.
- **Headers**:
  - `Authorization`: Authentication token obtained during registration.
- **Response**:
  - Status Code: 200 OK
  - Body: Returns a list of users who registered using the current user's referral code along with their registration timestamps.

## Usage

To use this API, follow the instructions for each endpoint as outlined above. Ensure that you include the required headers, such as the `Authorization` header with the authentication token, when making requests to authenticated endpoints.

## Dependencies

- FastAPI
- Pydantic
- Uvicorn

## Running the Application

To run the FastAPI application locally:

1. Install the dependencies using `pip install -r requirements.txt`.
2. Start the application using `uvicorn main:app --reload`.
3. Access the API at `http://localhost:8000`.

## Authors

- John Doe
- Jane Smith
