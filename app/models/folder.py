from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from app.database import Base


class Folder(Base):
    __tablename__ = "folders"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="folders")
    words = relationship("Word", back_populates="folder", cascade="all, delete-orphan")
    quiz_sessions = relationship("QuizSession", back_populates="folder", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Folder(id={self.id}, name='{self.name}', user_id={self.user_id})>"

    @property
    def word_count(self):
        """Get total number of words in this folder"""
        return len(self.words)

    @property
    def complete_words_count(self):
        """Get number of words with example sentences (quiz-ready words)"""
        return len([word for word in self.words if word.example_sentence])

    def can_start_quiz(self):
        """Check if folder has enough complete words for quiz"""
        from app.config import settings
        return self.complete_words_count >= settings.min_words_for_quiz

    def can_start_reading(self):
        """Check if folder has enough complete words for reading comprehension"""
        from app.config import settings
        return self.complete_words_count >= settings.min_words_for_reading