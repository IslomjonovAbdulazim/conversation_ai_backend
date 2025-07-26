from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timedelta
import jwt
import logging

from app.database import get_db
from app.models import User
from app.config import settings
from app.services.apple_auth import verify_apple_token

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()


# Pydantic models
class AppleSignInRequest(BaseModel):
    identity_token: str
    user_identifier: str
    nickname: str = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db)
):
    """Get current user from JWT token"""
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


@router.post("/apple-signin", response_model=AuthResponse)
async def apple_sign_in(
        request: AppleSignInRequest,
        db: Session = Depends(get_db)
):
    """Apple Sign In authentication"""
    try:
        # Verify Apple token
        apple_user_data = await verify_apple_token(request.identity_token)
        if not apple_user_data:
            raise HTTPException(status_code=400, detail="Invalid Apple token")

        email = apple_user_data.get("email")
        if not email:
            raise HTTPException(status_code=400, detail="Email not found in Apple token")

        # Find or create user
        user = db.query(User).filter(User.email == email).first()

        if user is None:
            # Create new user
            user = User(
                email=email,
                nickname=request.nickname or "User"
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"New user created: {email}")
        else:
            # Update nickname if provided
            if request.nickname and request.nickname != user.nickname:
                user.nickname = request.nickname
                db.commit()

        # Create access token
        access_token = create_access_token(data={"sub": user.id})

        return AuthResponse(
            access_token=access_token,
            user={
                "id": user.id,
                "email": user.email,
                "nickname": user.nickname,
                "created_at": user.created_at.isoformat()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Apple sign in error: {str(e)}")
        raise HTTPException(status_code=500, detail="Authentication failed")


@router.post("/test-login")
async def test_login(
        email: str,
        nickname: str = "Test User",
        db: Session = Depends(get_db)
):
    """Test login endpoint for development"""
    if not settings.debug:
        raise HTTPException(status_code=404, detail="Endpoint not available")

    # Basic email validation
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Invalid email format")

    # Find or create test user
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        user = User(email=email, nickname=nickname)
        db.add(user)
        db.commit()
        db.refresh(user)

    # Create access token
    access_token = create_access_token(data={"sub": user.id})

    return AuthResponse(
        access_token=access_token,
        user={
            "id": user.id,
            "email": user.email,
            "nickname": user.nickname,
            "created_at": user.created_at.isoformat()
        }
    )