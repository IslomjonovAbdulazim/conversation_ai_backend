from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional
import jwt
from datetime import datetime, timedelta
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
    nickname: Optional[str] = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserInfo(BaseModel):
    id: int
    email: str
    nickname: str
    created_at: datetime


# JWT functions
def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)
    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user_id"""
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )


def get_current_user(
        user_id: int = Depends(verify_token),
        db: Session = Depends(get_db)
):
    """Get current authenticated user"""
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user


@router.post("/apple-signin", response_model=AuthResponse)
async def apple_signin(
        request: AppleSignInRequest,
        db: Session = Depends(get_db)
):
    """
    Apple Sign In authentication
    """
    try:
        # Verify Apple identity token
        apple_user_data = await verify_apple_token(request.identity_token)

        if not apple_user_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Apple identity token"
            )

        email = apple_user_data.get("email")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not provided by Apple"
            )

        # Check if user exists
        user = db.query(User).filter(User.email == email).first()

        if user is None:
            # Create new user
            nickname = request.nickname or email.split("@")[0]  # Default nickname from email
            user = User(
                email=email,
                nickname=nickname
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            logger.info(f"New user created: {user.email}")
        else:
            # Update nickname if provided
            if request.nickname and request.nickname != user.nickname:
                user.nickname = request.nickname
                db.commit()
                logger.info(f"User nickname updated: {user.email}")

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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


# Test endpoint for development (remove in production)
@router.post("/test-login")
async def test_login(
        email: EmailStr,
        nickname: str = "Test User",
        db: Session = Depends(get_db)
):
    """
    Test login endpoint for development
    """
    if not settings.debug:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Endpoint not available"
        )

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