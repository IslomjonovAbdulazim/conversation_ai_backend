from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Dict
import logging

from app.database import get_db
from app.models import User, Folder, Word, QuizSession, WordStats
from app.routers.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models
class UserProfileResponse(BaseModel):
    user: Dict
    stats: Dict


class UpdateProfileRequest(BaseModel):
    nickname: str


class UserUpdateResponse(BaseModel):
    success: bool
    user: Dict


@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Get user profile with statistics
    """
    try:
        # Calculate user statistics
        total_folders = db.query(Folder).filter(Folder.user_id == current_user.id).count()

        total_words = db.query(Word).join(Folder).filter(
            Folder.user_id == current_user.id
        ).count()

        total_quizzes = db.query(QuizSession).filter(
            QuizSession.user_id == current_user.id
        ).count()

        # Get words by category
        word_stats = db.query(WordStats).filter(
            WordStats.user_id == current_user.id
        ).all()

        words_by_category = {
            "not_known": 0,
            "normal": 0,
            "strong": 0
        }

        for stat in word_stats:
            category = stat.category or "not_known"
            words_by_category[category] += 1

        return UserProfileResponse(
            user={
                "id": current_user.id,
                "email": current_user.email,
                "nickname": current_user.nickname,
                "created_at": current_user.created_at.isoformat()
            },
            stats={
                "total_folders": total_folders,
                "total_words": total_words,
                "total_quizzes": total_quizzes,
                "words_by_category": words_by_category
            }
        )

    except Exception as e:
        logger.error(f"Error getting user profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user profile"
        )


@router.put("/profile", response_model=UserUpdateResponse)
async def update_user_profile(
        request: UpdateProfileRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Update user profile (nickname)
    """
    try:
        # Validate nickname
        if not request.nickname or len(request.nickname.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nickname cannot be empty"
            )

        if len(request.nickname) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Nickname too long (max 50 characters)"
            )

        # Update user nickname
        current_user.nickname = request.nickname.strip()
        db.commit()
        db.refresh(current_user)

        logger.info(f"User {current_user.id} updated nickname to: {current_user.nickname}")

        return UserUpdateResponse(
            success=True,
            user={
                "id": current_user.id,
                "email": current_user.email,
                "nickname": current_user.nickname,
                "created_at": current_user.created_at.isoformat()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )