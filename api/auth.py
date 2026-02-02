from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "X-API-Key"
API_KEY = "my-secret-api-key"   # use your real key

api_key_header = APIKeyHeader(
    name=API_KEY_NAME,
    auto_error=False
)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
    return api_key

