from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import logging
from openai import OpenAI

from app.database import get_db
from app.models import User, Folder, Word
from app.routers.auth import get_current_user
from app.config import settings
from app.services.vision_service import vision_service

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize OpenAI client
client = OpenAI(api_key=settings.openai_api_key)


# Pydantic models
class AddWordRequest(BaseModel):
    word: str
    translation: str
    example_sentence: Optional[str] = None


class UpdateWordRequest(BaseModel):
    word: Optional[str] = None
    translation: Optional[str] = None
    example_sentence: Optional[str] = None


class WordResponse(BaseModel):
    id: int
    word: str
    translation: str
    example_sentence: Optional[str]
    created_at: str


class TranslationSuggestion(BaseModel):
    translation: str
    context: Optional[str] = None


class TranslationSuggestionsResponse(BaseModel):
    word: str
    suggestions: List[TranslationSuggestion]


class ExtractedWord(BaseModel):
    text: str
    confidence: float


class ImageExtractionResponse(BaseModel):
    extracted_words: List[ExtractedWord]
    success: bool


# Helper function to get folder or 404
def get_folder_or_404(folder_id: int, user_id: int, db: Session) -> Folder:
    folder = db.query(Folder).filter(
        Folder.id == folder_id,
        Folder.user_id == user_id
    ).first()
    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")
    return folder


async def get_translation_suggestions(english_word: str) -> List[TranslationSuggestion]:
    """Get multiple translation suggestions for an English word using OpenAI"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful English-Uzbek translator. Provide exactly 5 different translation options for English words. Format each as 'translation - context' where context explains when to use it."
                },
                {
                    "role": "user",
                    "content": f"Translate '{english_word}' to Uzbek. Give me 5 different options with context."
                }
            ],
            max_tokens=200,
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()
        suggestions = []

        # Parse the response
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove numbering if present
            line = line.lstrip('12345.- ')

            if ' - ' in line:
                parts = line.split(' - ', 1)
                translation = parts[0].strip()
                context = parts[1].strip()
                suggestions.append(TranslationSuggestion(
                    translation=translation,
                    context=context
                ))
            elif line:
                suggestions.append(TranslationSuggestion(
                    translation=line.strip(),
                    context=None
                ))

        # Ensure we have at least some suggestions
        if not suggestions:
            # Add basic fallbacks for common words
            word_lower = english_word.lower()
            if word_lower == "car":
                suggestions = [
                    TranslationSuggestion(translation="mashina", context="general vehicle"),
                    TranslationSuggestion(translation="avtomobil", context="formal term"),
                    TranslationSuggestion(translation="transport", context="means of transport"),
                ]
            elif word_lower == "house":
                suggestions = [
                    TranslationSuggestion(translation="uy", context="general house"),
                    TranslationSuggestion(translation="bino", context="building"),
                    TranslationSuggestion(translation="turar joy", context="dwelling place"),
                ]
            else:
                suggestions = [
                    TranslationSuggestion(translation=f"{english_word} (tarjima kerak)", context="needs translation"),
                ]

        return suggestions[:5]  # Return max 5 suggestions

    except Exception as e:
        logger.error(f"Error getting translation suggestions: {str(e)}")
        # Fallback suggestions
        return [
            TranslationSuggestion(translation="tarjima kerak", context="translation needed"),
            TranslationSuggestion(translation="ma'no", context="meaning"),
            TranslationSuggestion(translation="tushuncha", context="concept"),
        ]


async def extract_text_from_image(image_content: bytes) -> List[ExtractedWord]:
    """Extract text from image using Google Vision API"""
    try:
        words_data = vision_service.extract_text(image_content)

        extracted_words = []
        for word_data in words_data:
            extracted_words.append(ExtractedWord(
                text=word_data["text"],
                confidence=word_data["confidence"]
            ))

        return extracted_words

    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        return []


@router.post("/extract-image", response_model=ImageExtractionResponse)
async def extract_text_from_uploaded_image(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user)
):
    """Extract text from uploaded image"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )

        # Read image content
        image_content = await file.read()

        # Extract text
        extracted_words = await extract_text_from_image(image_content)

        return ImageExtractionResponse(
            extracted_words=extracted_words,
            success=True
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting image text: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to extract text from image"
        )


@router.post("/suggestions", response_model=TranslationSuggestionsResponse)
async def get_word_translation_suggestions(
        word: str,
        current_user: User = Depends(get_current_user)
):
    """Get multiple translation suggestions for an English word"""
    try:
        if not word or len(word.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Word cannot be empty"
            )

        suggestions = await get_translation_suggestions(word.strip())

        return TranslationSuggestionsResponse(
            word=word.strip(),
            suggestions=suggestions
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting translation suggestions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get translation suggestions"
        )


@router.post("/{folder_id}", response_model=WordResponse)
async def add_word(
        folder_id: int,
        request: AddWordRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Add word to folder"""
    try:
        # Get folder
        folder = get_folder_or_404(folder_id, current_user.id, db)

        # Validate input
        if not request.word or len(request.word.strip()) == 0:
            raise HTTPException(status_code=400, detail="Word cannot be empty")

        if not request.translation or len(request.translation.strip()) == 0:
            raise HTTPException(status_code=400, detail="Translation cannot be empty")

        # Check if word already exists in this folder
        existing_word = db.query(Word).filter(
            Word.folder_id == folder_id,
            Word.word.ilike(request.word.strip())
        ).first()

        if existing_word:
            raise HTTPException(
                status_code=400,
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
        db.commit()
        db.refresh(word)

        return WordResponse(
            id=word.id,
            word=word.word,
            translation=word.translation,
            example_sentence=word.example_sentence,
            created_at=word.created_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding word: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to add word")


@router.get("/{word_id}", response_model=WordResponse)
async def get_word_detail(
        word_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Get word details"""
    try:
        # Get word that belongs to user's folder
        word = db.query(Word).join(Folder).filter(
            Word.id == word_id,
            Folder.user_id == current_user.id
        ).first()

        if not word:
            raise HTTPException(status_code=404, detail="Word not found")

        return WordResponse(
            id=word.id,
            word=word.word,
            translation=word.translation,
            example_sentence=word.example_sentence,
            created_at=word.created_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting word details: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch word details")


@router.put("/{word_id}", response_model=WordResponse)
async def update_word(
        word_id: int,
        request: UpdateWordRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Update word"""
    try:
        # Get word that belongs to user's folder
        word = db.query(Word).join(Folder).filter(
            Word.id == word_id,
            Folder.user_id == current_user.id
        ).first()

        if not word:
            raise HTTPException(status_code=404, detail="Word not found")

        # Update fields if provided
        if request.word is not None:
            if not request.word or len(request.word.strip()) == 0:
                raise HTTPException(status_code=400, detail="Word cannot be empty")
            word.word = request.word.strip().lower()

        if request.translation is not None:
            if not request.translation or len(request.translation.strip()) == 0:
                raise HTTPException(status_code=400, detail="Translation cannot be empty")
            word.translation = request.translation.strip()

        if request.example_sentence is not None:
            word.example_sentence = request.example_sentence.strip() if request.example_sentence else None

        db.commit()
        db.refresh(word)

        return WordResponse(
            id=word.id,
            word=word.word,
            translation=word.translation,
            example_sentence=word.example_sentence,
            created_at=word.created_at.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating word: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update word")


@router.delete("/{word_id}")
async def delete_word(
        word_id: int,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Delete word"""
    try:
        # Get word that belongs to user's folder
        word = db.query(Word).join(Folder).filter(
            Word.id == word_id,
            Folder.user_id == current_user.id
        ).first()

        if not word:
            raise HTTPException(status_code=404, detail="Word not found")

        db.delete(word)
        db.commit()

        return {"success": True, "message": "Word deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting word: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete word")