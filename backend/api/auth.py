from datetime import datetime, timedelta
from typing import Optional
from jose import jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Security configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Change in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# In-memory user store (use database in production)
fake_users_db = {
    "demo@portfoliomax.com": {
        "email": "demo@portfoliomax.com",
        "hashed_password": pwd_context.hash("demo123"),
        "full_name": "Demo User",
        "saved_portfolios": {}  # saved_id -> portfolio_data
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get current authenticated user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.JWTError:
        raise credentials_exception
    
    user = fake_users_db.get(email)
    if user is None:
        raise credentials_exception
    return user

def register_user(email: str, password: str, full_name: str) -> dict:
    """Register a new user"""
    if email in fake_users_db:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(password)
    user_data = {
        "email": email,
        "hashed_password": hashed_password,
        "full_name": full_name
    }
    fake_users_db[email] = user_data
    return user_data

def authenticate_user(email: str, password: str) -> Optional[dict]:
    """Authenticate user credentials"""
    user = fake_users_db.get(email)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def get_user_saved_portfolios(email: str) -> dict:
    """Get all saved portfolios for a user"""
    user = fake_users_db.get(email)
    if not user:
        return {}
    return user.get("saved_portfolios", {})

def save_user_portfolio(email: str, saved_id: str, portfolio_data: dict) -> bool:
    """Save a portfolio for a user"""
    user = fake_users_db.get(email)
    if not user:
        return False
    
    if "saved_portfolios" not in user:
        user["saved_portfolios"] = {}
    
    user["saved_portfolios"][saved_id] = portfolio_data
    return True

def delete_user_portfolio(email: str, saved_id: str) -> bool:
    """Delete a saved portfolio for a user"""
    user = fake_users_db.get(email)
    if not user or "saved_portfolios" not in user:
        return False
    
    if saved_id in user["saved_portfolios"]:
        del user["saved_portfolios"][saved_id]
        return True
    return False

def update_user_portfolio(email: str, saved_id: str, updates: dict) -> bool:
    """Update metadata for a saved portfolio"""
    user = fake_users_db.get(email)
    if not user or "saved_portfolios" not in user:
        return False
    
    if saved_id not in user["saved_portfolios"]:
        return False
    
    portfolio = user["saved_portfolios"][saved_id]
    for key, value in updates.items():
        if value is not None:
            portfolio[key] = value
    
    return True 