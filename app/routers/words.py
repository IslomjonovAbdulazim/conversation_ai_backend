from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import time

from app.database import get_db
from app.models import User, Folder, Word, WordStats
from app.routers.auth import get_current_user
from app.services.google_vision import extract_text_from_image
from app.services.openai_service import translate_to_uzbek, generate_example_sentence

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models
class AddWordRequest(BaseModel):
    word: str
    translation: str
    example_sentence: Optional[str] = None


class WordResponse(BaseModel):
    id: int
    word: str
    translation: str
    example_sentence: Optional[str]
    added_at: str


class AddWordResponse(BaseModel):
    word: WordResponse
    stats: Dict


class ExtractedWord(BaseModel):
    word: str
    translation: str
    confidence: float


class OCRResponse(BaseModel):
    extracted_words: List[ExtractedWord]
    total_extracted: int
    processing_time: float


class BulkAddRequest(BaseModel):
    words: List[AddWordRequest]


class BulkAddResponse(BaseModel):
    success: bool
    added_count: int
    words: List[WordResponse]


class BulkDeleteRequest(BaseModel):
    word_ids: List[int]


class BulkDeleteResponse(BaseModel):
    success: bool
    deleted_count: int
    deleted_words: List[Dict]


class GenerateExampleRequest(BaseModel):
    word: str
    translation: str


class GenerateExampleResponse(BaseModel):
    example_sentence: str
    alternatives: List[str] = []


class WordDetailResponse(BaseModel):
    word: Dict
    stats: Dict


class UpdateWordRequest(BaseModel):
    word: Optional[str] = None
    translation: Optional[str] = None
    example_sentence: Optional[str] = None


class UpdateWordResponse(BaseModel):
    success: bool
    word: Dict


class DeleteWordResponse(BaseModel):
    success: bool
    message: str
    deleted_word: Dict


def get_folder_or_404(folder_id: int, user_id: int, db: Session):
    """Helper function to get folder or raise 404"""
    folder = db.query(Folder).filter(
        Folder.id == folder_id,
        Folder.user_id == user_id
    ).first()

    if not folder:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Folder not found"
        )

    return folder


def create_word_stats(word_id: int, user_id: int, db: Session):
    """Create initial word stats for user"""
    word_stats = WordStats(
        word_id=word_id,
        user_id=user_id,
        category="not_known",
        last_5_results=[],
        total_attempts=0,
        correct_attempts=0
    )
    db.add(word_stats)
    return word_stats


# SPECIFIC ROUTES FIRST - these must come before parameterized routes

@router.post("/upload-photo", response_model=OCRResponse)
async def upload_photo_ocr(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user)
):
    """
    Extract text from image using Google Vision API
    """
    try:
        start_time = time.time()

        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )

        # Read file content
        file_content = await file.read()

        # Extract text using Google Vision
        extracted_text = extract_text_from_image(file_content)

        if not extracted_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text found in image"
            )

        # Split into words and translate
        words = [word.strip() for word in extracted_text.split() if len(word.strip()) > 2]

        if not words:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid words found in image"
            )

        # Translate words to Uzbek
        translated_words = []
        for word in words[:20]:  # Limit to 20 words
            try:
                translation = await translate_to_uzbek(word)
                if translation:
                    translated_words.append(ExtractedWord(
                        word=word.lower(),
                        translation=translation,
                        confidence=0.8  # Could be improved with actual confidence scores
                    ))
            except Exception as e:
                logger.warning(f"Failed to translate word '{word}': {str(e)}")
                continue

        processing_time = round(time.time() - start_time, 2)

        logger.info(f"OCR processed {len(translated_words)} words in {processing_time}s for user {current_user.id}")

        return OCRResponse(
            extracted_words=translated_words,
            total_extracted=len(translated_words),
            processing_time=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process image"
        )


@router.post("/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_words(
        request: BulkDeleteRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Delete multiple words at once
    """
    try:
        if not request.word_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No word IDs provided"
            )

        # Get words that belong to user's folders
        words = db.query(Word).join(Folder).filter(
            Word.id.in_(request.word_ids),
            Folder.user_id == current_user.id
        ).all()

        if not words:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No words found to delete"
            )

        deleted_words = []
        for word in words:
            deleted_words.append({
                "id": word.id,
                "word": word.word
            })
            db.delete(word)

        db.commit()

        logger.info(f"Bulk deleted {len(deleted_words)} words by user {current_user.id}")

        return BulkDeleteResponse(
            success=True,
            deleted_count=len(deleted_words),
            deleted_words=deleted_words
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk deleting words: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete words"
        )


@router.post("/generate-example", response_model=GenerateExampleResponse)
async def generate_example(
        request: GenerateExampleRequest,
        current_user: User = Depends(get_current_user)
):
    """
    Generate example sentence for English word + Uzbek translation
    """
    try:
        if not request.word or not request.translation:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both word and translation are required"
            )

        example_sentence = await generate_example_sentence(
            request.word.strip(),
            request.translation.strip()
        )

        if not example_sentence:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate example sentence"
            )

        logger.info(f"Generated example for word '{request.word}' by user {current_user.id}")

        return GenerateExampleResponse(
            example_sentence=example_sentence,
            alternatives=[]  # Could add multiple examples in future
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating example: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate example sentence"
        )


# PARAMETERIZED ROUTES - these must come after specific routes

@router.post("/{folder_id}", response_model=AddWordResponse)
async def add_word(
        folder_id: int,
        request: AddWordRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Add word manually to folder
    """
    try:
        # Get folder
        folder = get_folder_or_404(folder_id, current_user.id, db)

        # Validate input
        if not request.word or len(request.word.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Word cannot be empty"
            )

        if not request.translation or len(request.translation.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Translation cannot be empty"
            )

        # Check if word already exists in this folder
        existing_word = db.query(Word).filter(
            Word.folder_id == folder_id,
            Word.word.ilike(request.word.strip())
        ).first()

        if existing_word:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Word already exists in this folder"
            )

        # Create new word
        word = Word(
            folder_id=folder_id,
            word=request.word.strip().lower(),
            translation=request.translation.strip(),
            example_sentence=request.example_sentence.strip() if request.example_sentence else None
        )

        db.add(word)
        db.flush()  # Get word.id

        # Create word stats
        word_stats = create_word_stats(word.id, current_user.id, db)

        db.commit()
        db.refresh(word)

        logger.info(f"Word '{word.word}' added to folder {folder_id} by user {current_user.id}")

        return AddWordResponse(
            word=WordResponse(
                id=word.id,
                word=word.word,
                translation=word.translation,
                example_sentence=word.example_sentence,
                added_at=word.added_at.isoformat()
            ),
            stats={
                "category": word_stats.category,
                "accuracy": 0,
                "total_attempts": 0
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding word: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add word"
        )


@router.post("/{folder_id}/bulk-add", response_model=BulkAddResponse)
async def bulk_add_words(
        folder_id: int,
        request: BulkAddRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Save multiple words at once
    """
    try:
        # Get folder
        folder = get_folder_or_404(folder_id, current_user.id, db)

        if not request.words:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No words provided"
            )

        added_words = []

        for word_data in request.words:
            # Validate word
            if not word_data.word or not word_data.translation:
                continue  # Skip invalid words

            # Check if word already exists
            existing_word = db.query(Word).filter(
                Word.folder_id == folder_id,
                Word.word.ilike(word_data.word.strip())
            ).first()

            if existing_word:
                continue  # Skip duplicates

            # Create word
            word = Word(
                folder_id=folder_id,
                word=word_data.word.strip().lower(),
                translation=word_data.translation.strip(),
                example_sentence=word_data.example_sentence.strip() if word_data.example_sentence else None
            )

            db.add(word)
            db.flush()  # Get word.id

            # Create word stats
            create_word_stats(word.id, current_user.id, db)

            added_words.append(WordResponse(
                id=word.id,
                word=word.word,
                translation=word.translation,
                example_sentence=word.example_sentence,
                added_at=word.added_at.isoformat()
            ))

        db.commit()

        logger.info(f"Bulk added {len(added_words)} words to folder {folder_id} by user {current_user.id}")

        return BulkAddResponse(
            success=True,
            added_count=len(added_words),
            words=added_words
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk adding words: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add words"
        )


@router.get("/{word_id}", response_model=WordDetailResponse)
async def get_word_details(
        word_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Get word details with stats
    """
    try:
        # Get word that belongs to user's folder
        word = db.query(Word).join(Folder).filter(
            Word.id == word_id,
            Folder.user_id == current_user.id
        ).first()

        if not word:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Word not found"
            )

        # Get word stats
        word_stats = word.get_user_stats(current_user.id)

        return WordDetailResponse(
            word={
                "id": word.id,
                "word": word.word,
                "translation": word.translation,
                "example_sentence": word.example_sentence,
                "added_at": word.added_at.isoformat(),
                "folder": {
                    "id": word.folder.id,
                    "name": word.folder.name
                }
            },
            stats={
                "category": word_stats.category if word_stats else "not_known",
                "last_5_results": word_stats.last_5_results if word_stats else [],
                "total_attempts": word_stats.total_attempts if word_stats else 0,
                "correct_attempts": word_stats.correct_attempts if word_stats else 0,
                "accuracy": word_stats.accuracy if word_stats else 0,
                "last_quiz_date": None  # Could track this in future
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting word details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch word details"
        )


@router.put("/{word_id}", response_model=UpdateWordResponse)
async def update_word(
        word_id: int,
        request: UpdateWordRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Edit word/translation/example
    """
    try:
        # Get word that belongs to user's folder
        word = db.query(Word).join(Folder).filter(
            Word.id == word_id,
            Folder.user_id == current_user.id
        ).first()

        if not word:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Word not found"
            )

        # Update fields if provided
        if request.word is not None:
            if not request.word or len(request.word.strip()) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Word cannot be empty"
                )
            word.word = request.word.strip().lower()

        if request.translation is not None:
            if not request.translation or len(request.translation.strip()) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Translation cannot be empty"
                )
            word.translation = request.translation.strip()

        if request.example_sentence is not None:
            word.example_sentence = request.example_sentence.strip() if request.example_sentence else None

        db.commit()
        db.refresh(word)

        logger.info(f"Word {word.id} updated by user {current_user.id}")

        return UpdateWordResponse(
            success=True,
            word={
                "id": word.id,
                "word": word.word,
                "translation": word.translation,
                "example_sentence": word.example_sentence,
                "added_at": word.added_at.isoformat()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating word: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update word"
        )


@router.delete("/{word_id}", response_model=DeleteWordResponse)
async def delete_word(
        word_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """
    Delete word
    """
    try:
        # Get word that belongs to user's folder
        word = db.query(Word).join(Folder).filter(
            Word.id == word_id,
            Folder.user_id == current_user.id
        ).first()

        if not word:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Word not found"
            )

        word_data = {
            "id": word.id,
            "word": word.word,
            "translation": word.translation
        }

        db.delete(word)
        db.commit()

        logger.info(f"Word '{word_data['word']}' deleted by user {current_user.id}")

        return DeleteWordResponse(
            success=True,
            message="Word deleted successfully",
            deleted_word=word_data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting word: {str(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete word"
        )