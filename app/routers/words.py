from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import logging
import openai

from app.database import get_db
from app.models import User, Folder, Word
from app.routers.auth import get_current_user
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Set OpenAI API key
openai.api_key = settings.openai_api_key


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


class UpdateWordTranslationRequest(BaseModel):
    translation: str


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
        response = await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful English-Uzbek translator. Provide multiple translation options for English words with brief context when useful. Return exactly 5 different translations."
                },
                {
                    "role": "user",
                    "content": f"Translate the English word '{english_word}' to Uzbek. Provide 5 different translation options with brief context if the word has multiple meanings. Format as: translation1 (context if needed), translation2 (context if needed), etc."
                }
            ],
            max_tokens=200,
            temperature=0.3
        )

        content = response.choices[0].message.content.strip()
        suggestions = []

        # Parse the response and extract translations
        lines = content.split('\n')
        for line in lines[:5]:  # Take only first 5
            line = line.strip()
            if line and not line.startswith(('1.', '2.', '3.', '4.', '5.')):
                # Remove numbering if present
                line = line.lstrip('12345.- ')

            if '(' in line and ')' in line:
                # Extract translation and context
                translation = line.split('(')[0].strip()
                context = line.split('(')[1].split(')')[0].strip()
                suggestions.append(TranslationSuggestion(
                    translation=translation,
                    context=context
                ))
            else:
                # Just translation without context
                if line:
                    suggestions.append(TranslationSuggestion(
                        translation=line.strip(),
                        context=None
                    ))

        # If parsing failed, provide fallback
        if not suggestions:
            suggestions = [
                TranslationSuggestion(translation="tarjima", context="general translation"),
                TranslationSuggestion(translation="ma'no", context="meaning/sense"),
                TranslationSuggestion(translation="tushuncha", context="concept"),
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


@router.put("/{word_id}/translation", response_model=WordResponse)
async def update_word_translation(
        word_id: int,
        request: UpdateWordTranslationRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    """Update only the translation of a word (for when user chooses from suggestions)"""
    try:
        # Get word that belongs to user's folder
        word = db.query(Word).join(Folder).filter(
            Word.id == word_id,
            Folder.user_id == current_user.id
        ).first()

        if not word:
            raise HTTPException(status_code=404, detail="Word not found")

        if not request.translation or len(request.translation.strip()) == 0:
            raise HTTPException(status_code=400, detail="Translation cannot be empty")

        # Update only the translation
        word.translation = request.translation.strip()
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
        logger.error(f"Error updating word translation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update word translation")


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