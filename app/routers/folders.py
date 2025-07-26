from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging

from app.database import get_db
from app.models import User, Folder, Word, WordStats
from app.routers.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models
class CreateFolderRequest(BaseModel):
    name: str
    description: Optional[str] = None


class UpdateFolderRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class FolderResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    word_count: int
    created_at: str


class FolderListResponse(BaseModel):
    folders: List[FolderResponse]
    total_count: int


class FolderDetailResponse(BaseModel):
    folder: Dict
    words: List[Dict]


class DeleteFolderResponse(BaseModel):
    success: bool
    message: str
    deleted_words_count: int


@router.get("", response_model=FolderListResponse)
async def get_folders(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Get all folders for current user
    """
    try:
        folders = db.query(Folder).filter(
            Folder.user_id == current_user.id
        ).order_by(Folder.created_at.desc()).all()

        folder_list = []
        for folder in folders:
            folder_list.append(FolderResponse(
                id=folder.id,
                name=folder.name,
                description=folder.description,
                word_count=folder.word_count,
                created_at=folder.created_at.isoformat()
            ))

        return FolderListResponse(
            folders=folder_list,
            total_count=len(folder_list)
        )

    except Exception as e:
        logger.error(f"Error getting folders: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch folders"
        )


@router.post("", response_model=FolderResponse)
async def create_folder(
        request: CreateFolderRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Create new folder
    """
    try:
        # Validate input
        if not request.name or len(request.name.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Folder name cannot be empty"
            )

        if len(request.name) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Folder name too long (max 100 characters)"
            )

        # Check if folder name already exists for this user
        existing_folder = db.query(Folder).filter(
            Folder.user_id == current_user.id,
            Folder.name == request.name.strip()
        ).first()

        if existing_folder:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Folder with this name already exists"
            )

        # Create new folder
        folder = Folder(
            user_id=current_user.id,
            name=request.name.strip(),
            description=request.description.strip() if request.description else None
        )

        db.add(folder)
        db.commit()
        db.refresh(folder)

        logger.info(f"Folder created: {folder.name} by user {current_user.id}")

        return FolderResponse(
            id=folder.id,
            name=folder.name,
            description=folder.description,
            word_count=0,
            created_at=folder.created_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating folder: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create folder"
        )


@router.get("/{folder_id}", response_model=FolderDetailResponse)
async def get_folder_details(
        folder_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Get folder details with words
    """
    try:
        # Get folder
        folder = db.query(Folder).filter(
            Folder.id == folder_id,
            Folder.user_id == current_user.id
        ).first()

        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        # Get words with stats
        words_data = []
        for word in folder.words:
            # Get user stats for this word
            word_stats = word.get_user_stats(current_user.id)

            word_data = {
                "id": word.id,
                "word": word.word,
                "translation": word.translation,
                "example_sentence": word.example_sentence,
                "added_at": word.added_at.isoformat(),
                "stats": {
                    "category": word_stats.category if word_stats else "not_known",
                    "accuracy": word_stats.accuracy if word_stats else 0
                }
            }
            words_data.append(word_data)

        # Sort words by added_at desc
        words_data.sort(key=lambda x: x["added_at"], reverse=True)

        return FolderDetailResponse(
            folder={
                "id": folder.id,
                "name": folder.name,
                "description": folder.description,
                "word_count": folder.word_count,
                "created_at": folder.created_at.isoformat()
            },
            words=words_data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting folder details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch folder details"
        )


@router.put("/{folder_id}", response_model=FolderResponse)
async def update_folder(
        folder_id: int,
        request: UpdateFolderRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Update folder name and/or description
    """
    try:
        # Get folder
        folder = db.query(Folder).filter(
            Folder.id == folder_id,
            Folder.user_id == current_user.id
        ).first()

        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        # Update fields if provided
        if request.name is not None:
            if not request.name or len(request.name.strip()) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Folder name cannot be empty"
                )

            if len(request.name) > 100:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Folder name too long (max 100 characters)"
                )

            # Check if new name conflicts with existing folders
            existing_folder = db.query(Folder).filter(
                Folder.user_id == current_user.id,
                Folder.name == request.name.strip(),
                Folder.id != folder_id
            ).first()

            if existing_folder:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Folder with this name already exists"
                )

            folder.name = request.name.strip()

        if request.description is not None:
            folder.description = request.description.strip() if request.description else None

        db.commit()
        db.refresh(folder)

        logger.info(f"Folder {folder.id} updated by user {current_user.id}")

        return FolderResponse(
            id=folder.id,
            name=folder.name,
            description=folder.description,
            word_count=folder.word_count,
            created_at=folder.created_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating folder: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update folder"
        )


@router.delete("/{folder_id}", response_model=DeleteFolderResponse)
async def delete_folder(
        folder_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Delete folder and all associated words
    """
    try:
        # Get folder
        folder = db.query(Folder).filter(
            Folder.id == folder_id,
            Folder.user_id == current_user.id
        ).first()

        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )

        # Count words for response
        word_count = folder.word_count
        folder_name = folder.name

        # Delete folder (cascade will delete words and related data)
        db.delete(folder)
        db.commit()

        logger.info(f"Folder '{folder_name}' deleted by user {current_user.id}, {word_count} words removed")

        return DeleteFolderResponse(
            success=True,
            message="Folder and all associated words deleted successfully",
            deleted_words_count=word_count
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting folder: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete folder"
        )