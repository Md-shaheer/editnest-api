from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from PIL import Image
import io
import re
import hashlib
import os

from database import get_db, create_tables
from auth import (
    create_user, authenticate_user, create_access_token,
    get_user_by_email, get_user_by_username, decode_token
)

app = FastAPI(title="EditNest API", version="1.0.0")

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

create_tables()
# Pre-download the rembg model on startup
from rembg import new_session, remove

print("Pre-loading AI model...")
model_session = new_session("u2net")
print("AI model loaded!")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://editnest-production.up.railway.app",
        "https://glistening-serenity-production.up.railway.app",
        "https://editnest-theta.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_FILE_SIZE = 10 * 1024 * 1024

# --- Pydantic Models ---
class SignupRequest(BaseModel):
    email: str
    username: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

# --- Auth Routes ---
@app.post("/auth/signup")
def signup(data: SignupRequest, db: Session = Depends(get_db)):
    if get_user_by_email(db, data.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    if get_user_by_username(db, data.username):
        raise HTTPException(status_code=400, detail="Username already taken")
    if len(data.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    if not re.search(r"[A-Za-z]", data.password) or not re.search(r"[0-9]", data.password):
        raise HTTPException(status_code=400, detail="Password must contain at least one letter and one number")
    user = create_user(db, data.email, data.username, data.password)
    token = create_access_token({"sub": user.email})
    return {"token": token, "username": user.username, "email": user.email}

@app.post("/auth/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, data.email, data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": user.email})
    return {"token": token, "username": user.username, "email": user.email}

@app.get("/auth/me")
def get_me(authorization: str = Header(None), db: Session = Depends(get_db)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    email = decode_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = get_user_by_email(db, email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"username": user.username, "email": user.email}

# --- Image Routes ---
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/remove-bg")
async def remove_background(
    file: UploadFile = File(...),
    authorization: str = Header(None),
    db: Session = Depends(get_db)
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Please login to use this feature")
    token = authorization.split(" ")[1]
    email = decode_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type")
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

    # Check if the image has already been processed
    file_hash = hashlib.sha256(contents).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{file_hash}.png")
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return Response(
                content=f.read(),
                media_type="image/png",
                headers={"Content-Disposition": "attachment; filename=removed_bg.png"},
            )

    try:
        input_image = Image.open(io.BytesIO(contents))
        input_image.verify()
        output_bytes = remove(contents, session=model_session)
        output_image = Image.open(io.BytesIO(output_bytes))
        png_buffer = io.BytesIO()
        output_image.save(png_buffer, format="PNG", optimize=True, compress_level=9)
        png_data = png_buffer.getvalue()
        
        # Save the result to the cache
        with open(cache_path, "wb") as f:
            f.write(png_data)
            
        return Response(
            content=png_data,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=removed_bg.png"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
