from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import logging

from app.database import get_db
from app.models import User, Folder, Word
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


class WordResponse(BaseModel):
    id: int
    word: str
    translation: str
    example_sentence: Optional[str]
    created_at: str


class FolderDetailResponse(BaseModel):
    folder: FolderResponse
    words: List[WordResponse]


@router.get("", response_model=FolderListResponse)
async def get_folders(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all folders for current user"""
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
        raise HTTPException(status_code=500, detail="Failed to fetch folders")


@router.post("", response_model=FolderResponse)
async def create_folder(
    request: CreateFolderRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create new folder"""
    try:
        # Check if folder name already exists for this user
        existing_folder = db.query(Folder).filter(
            Folder.user_id == current_user.id,
            Folder.name.ilike(request.name.strip())
        ).first()

        if existing_folder:
            raise HTTPException(
                status_code=400,
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
        raise HTTPException(status_code=500, detail="Failed to create folder")


@router.get("/{folder_id}", response_model=FolderDetailResponse)
async def get_folder_detail(
    folder_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get folder details with words"""
    try:
        # Get folder that belongs to user
        folder = db.query(Folder).filter(
            Folder.id == folder_id,
            Folder.user_id == current_user.id
        ).first()

        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")

        # Get words in folder
        words = db.query(Word).filter(
            Word.folder_id == folder_id
        ).order_by(Word.created_at.desc()).all()

        word_list = []
        for word in words:
            word_list.append(WordResponse(
                id=word.id,
                word=word.word,
                translation=word.translation,
                example_sentence=word.example_sentence,
                created_at=word.created_at.isoformat()
            ))

        return FolderDetailResponse(
            folder=FolderResponse(
                id=folder.id,
                name=folder.name,
                description=folder.description,
                word_count=len(word_list),
                created_at=folder.created_at.isoformat()
            ),
            words=word_list
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting folder detail: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch folder details")


@router.put("/{folder_id}", response_model=FolderResponse)
async def update_folder(
    folder_id: int,
    request: UpdateFolderRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update folder"""
    try:
        # Get folder that belongs to user
        folder = db.query(Folder).filter(
            Folder.id == folder_id,
            Folder.user_id == current_user.id
        ).first()

        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")

        # Update fields if provided
        if request.name is not None:
            folder.name = request.name.strip()
        if request.description is not None:
            folder.description = request.description.strip() if request.description else None

        db.commit()
        db.refresh(folder)

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
        raise HTTPException(status_code=500, detail="Failed to update folder")


@router.delete("/{folder_id}")
async def delete_folder(
    folder_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete folder and all its words"""
    try:
        # Get folder that belongs to user
        folder = db.query(Folder).filter(
            Folder.id == folder_id,
            Folder.user_id == current_user.id
        ).first()

        if not folder:
            raise HTTPException(status_code=404, detail="Folder not found")

        word_count = folder.word_count
        db.delete(folder)
        db.commit()

        return {
            "success": True,
            "message": "Folder deleted successfully",
            "deleted_words_count": word_count
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting folder: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete folder")