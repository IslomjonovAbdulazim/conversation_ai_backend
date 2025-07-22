# Vocabulary Learning App - Complete Technical Documentation

A comprehensive English-Uzbek vocabulary learning platform with AI-powered features, gamification, and voice conversation capabilities.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture & Tech Stack](#architecture--tech-stack)
3. [Database Models & Relationships](#database-models--relationships)
4. [API Endpoints Documentation](#api-endpoints-documentation)
5. [Data Flow Architecture](#data-flow-architecture)
6. [Critical Algorithms & Business Logic](#critical-algorithms--business-logic)
7. [External Service Integrations](#external-service-integrations)
8. [Authentication & Security](#authentication--security)
9. [Quiz Engine Deep Dive](#quiz-engine-deep-dive)
10. [Voice Conversation System](#voice-conversation-system)
11. [Setup & Deployment](#setup--deployment)
12. [Performance & Scalability](#performance--scalability)

---

## System Overview

### Core Concept
The Vocabulary Learning App is a sophisticated language learning platform that combines traditional vocabulary study methods with modern AI technologies. Users create folders to organize English words with Uzbek translations, learn through various quiz types, and practice conversational skills with AI tutors.

### Key Features
- **Intelligent Folder Organization**: Topic-based word management
- **OCR Photo Processing**: Extract words from images using Google Vision API
- **Adaptive Learning System**: Smart word categorization based on performance
- **Multiple Quiz Types**: Anagram, Translation Blitz, Word Blitz, Reading Comprehension
- **AI Voice Conversations**: Topic-specific tutors (Cars, Football, Travel)
- **Apple Authentication**: Secure sign-in with email-based accounts
- **Real-time Progress Tracking**: Last 5 quiz results determine word difficulty

---

## Architecture & Tech Stack

### Backend Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
├─────────────────────────────────────────────────────────────┤
│  Authentication │  Routers  │  Business Logic │  Utilities  │
│     Layer       │   Layer   │      Layer      │    Layer    │
├─────────────────────────────────────────────────────────────┤
│                    Service Layer                            │
│  Apple Auth │ Google Vision │ OpenAI │ ElevenLabs │ Utils   │
├─────────────────────────────────────────────────────────────┤
│                    Data Layer                               │
│              SQLAlchemy ORM + PostgreSQL                   │
└─────────────────────────────────────────────────────────────┘
```

### Technology Stack
- **Web Framework**: FastAPI 0.104.1 (High-performance async Python)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Authentication**: Apple Sign In with JWT tokens
- **AI Services**: 
  - OpenAI GPT-4o-mini (Translation & Example Generation)
  - Google Vision API (OCR Text Extraction)
  - ElevenLabs Conversational AI (Voice Tutoring)
- **Deployment**: Railway (PostgreSQL hosting)
- **Security**: JWT, bcrypt, cryptography

---

## Database Models & Relationships

### Entity Relationship Diagram
```
User (1) ──┬── (1:n) Folder (1) ── (1:n) Word (1) ── (1:1) WordStats
           │
           ├── (1:n) QuizSession (1) ── (1:n) QuizResult (n:1) ── Word
           │
           └── (1:n) WordStats

VoiceAgent (standalone - no direct relationships)
```

### Model Details

#### 1. User Model
```python
class User(Base):
    id: int (PK, Auto-increment)
    email: str (Unique, Not Null, Index)
    nickname: str (Not Null, Max 50 chars)
    created_at: datetime (Auto-generated)
    
    # Relationships
    folders: List[Folder] (CASCADE DELETE)
    word_stats: List[WordStats] (CASCADE DELETE)
    quiz_sessions: List[QuizSession] (CASCADE DELETE)
```

**Purpose**: Central user management with email-based authentication via Apple Sign In
**Key Features**: 
- Email uniqueness ensures one account per Apple ID
- Cascade deletion removes all user data when account deleted
- Created timestamp for user analytics

#### 2. Folder Model
```python
class Folder(Base):
    id: int (PK, Auto-increment)
    user_id: int (FK -> User.id, Not Null)
    name: str (Not Null, Max 100 chars)
    description: str (Nullable, Max 500 chars)
    created_at: datetime (Auto-generated)
    
    # Relationships
    user: User (Back reference)
    words: List[Word] (CASCADE DELETE)
    quiz_sessions: List[QuizSession] (CASCADE DELETE)
    
    # Computed Properties
    @property
    def word_count(self) -> int
    @property  
    def complete_words_count(self) -> int  # Words with example sentences
    
    # Business Logic Methods
    def can_start_quiz(self) -> bool      # >= 5 complete words
    def can_start_reading(self) -> bool   # >= 8 complete words
```

**Purpose**: Organize vocabulary by topics/categories (e.g., "Business English", "Travel Vocabulary")
**Key Features**:
- User isolation (folders belong to specific users)
- Quiz eligibility validation based on complete words
- Soft constraints on folder naming (user-level uniqueness recommended)

#### 3. Word Model
```python
class Word(Base):
    id: int (PK, Auto-increment)
    folder_id: int (FK -> Folder.id, Not Null)
    word: str (English word, Not Null, Lowercase stored)
    translation: str (Uzbek translation, Not Null)
    example_sentence: str (Nullable, English example)
    added_at: datetime (Auto-generated)
    
    # Relationships
    folder: Folder (Back reference)
    word_stats: List[WordStats] (CASCADE DELETE)
    quiz_results: List[QuizResult] (CASCADE DELETE)
    
    # Computed Properties
    @property
    def is_complete(self) -> bool  # Has non-empty example_sentence
    
    # Business Logic Methods
    def get_user_stats(self, user_id: int) -> WordStats
```

**Purpose**: Core vocabulary storage with English-Uzbek pairs
**Critical Business Rules**:
- Words without example sentences (`is_complete = False`) are excluded from quizzes
- English words stored in lowercase for consistent matching
- Example sentences required for meaningful quiz questions
- Translation required for all learning modes

#### 4. WordStats Model
```python
class WordStats(Base):
    id: int (PK, Auto-increment)
    word_id: int (FK -> Word.id, Not Null)
    user_id: int (FK -> User.id, Not Null)
    category: str (Default: "not_known")  # not_known | normal | strong
    last_5_results: JSON[List[bool]] (Default: [])
    total_attempts: int (Default: 0)
    correct_attempts: int (Default: 0)
    
    # Relationships
    word: Word (Back reference)
    user: User (Back reference)
    
    # Computed Properties
    @property
    def accuracy(self) -> int  # Percentage (0-100)
    
    # Critical Business Logic
    def add_result(self, is_correct: bool):
        # Update last_5_results (maintain exactly 5 items)
        # Increment counters
        # Auto-update category
    
    def update_category(self):
        # Algorithm: Based on last_5_results accuracy
        # <50% → not_known, 50-80% → normal, >80% → strong
```

**Purpose**: Track individual word learning progress per user
**Critical Algorithm**: Adaptive learning categorization based on recent performance
**Key Features**:
- Maintains exactly last 5 quiz results for recent performance assessment
- Auto-categorization triggers after each quiz attempt
- User-specific stats (same word can have different stats per user)

#### 5. QuizSession Model
```python
class QuizSession(Base):
    id: int (PK, Auto-increment)
    user_id: int (FK -> User.id, Not Null)
    folder_id: int (FK -> Folder.id, Not Null)
    quiz_type: str (anagram | translation_blitz | word_blitz | reading)
    duration: int (Total seconds, Nullable until completion)
    created_at: datetime (Auto-generated)
    
    # Relationships
    user: User (Back reference)
    folder: Folder (Back reference)
    quiz_results: List[QuizResult] (CASCADE DELETE)
    
    # Computed Properties
    @property
    def total_questions(self) -> int
    @property
    def correct_answers(self) -> int
    @property
    def accuracy(self) -> int
```

**Purpose**: Track quiz sessions with performance metrics
**Business Logic**: 
- Session created when quiz starts
- Duration calculated from individual question times
- Links to specific folder for targeted learning

#### 6. QuizResult Model
```python
class QuizResult(Base):
    id: int (PK, Auto-increment)
    session_id: int (FK -> QuizSession.id, Not Null)
    word_id: int (FK -> Word.id, Not Null)
    is_correct: bool (Quiz answer correctness)
    time_taken: int (Seconds for this question)
    
    # Relationships
    session: QuizSession (Back reference)
    word: Word (Back reference)
```

**Purpose**: Granular tracking of individual quiz question performance
**Analytics Use**: Time-per-question, word difficulty analysis, learning patterns

#### 7. VoiceAgent Model
```python
class VoiceAgent(Base):
    id: int (PK, Auto-increment)
    topic: str (cars | football | travel)
    title: str (Display name, e.g., "Car Expert")
    description: str (Topic description)
    image_url: str (Avatar/icon URL)
    agent_id: str (ElevenLabs agent identifier)
    is_active: bool (Default: True)
    created_at: datetime (Auto-generated)
```

**Purpose**: Configuration for AI voice conversation topics
**Business Logic**: 
- No direct relationship to users (global agents)
- ElevenLabs integration via agent_id
- Topic-specific conversation prompts managed in ElevenLabs

---

## API Endpoints Documentation

### Authentication Endpoints

#### POST /auth/apple-signin
**Purpose**: Authenticate user via Apple Sign In and create/login account

**Request Flow**:
```json
{
  "identity_token": "eyJhbGciOiJSUzI1NiIsImtpZCI...",
  "user_identifier": "000123.abc123def456.1234",
  "nickname": "John Doe"  // Optional on first login
}
```

**Processing Logic**:
1. Verify Apple identity token with Apple's public keys
2. Extract email from verified token payload
3. Check if user exists in database by email
4. If new user: Create User record with email + nickname
5. If existing user: Update nickname if provided
6. Generate JWT access token with user ID
7. Return token + user data

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "email": "user@example.com",
    "nickname": "John Doe",
    "created_at": "2025-01-15T10:30:00Z"
  }
}
```

**Security Features**:
- Apple token verification with RSA256 signature
- JWT token with 30-day expiration
- Email uniqueness enforcement
- No password storage required

---

### User Management Endpoints

#### GET /user/profile
**Purpose**: Retrieve user profile with learning statistics

**Authentication**: Bearer JWT token required

**Processing Logic**:
1. Verify JWT token and extract user_id
2. Fetch user from database
3. Calculate statistics:
   - Count folders (total_folders)
   - Count words across all folders (total_words)
   - Count completed quiz sessions (total_quizzes)
   - Aggregate word stats by category (words_by_category)

**Response Example**:
```json
{
  "user": {
    "id": 1,
    "email": "user@example.com",
    "nickname": "John Doe",
    "created_at": "2025-01-15T10:30:00Z"
  },
  "stats": {
    "total_folders": 5,
    "total_words": 150,
    "total_quizzes": 25,
    "words_by_category": {
      "not_known": 45,
      "normal": 80,
      "strong": 25
    }
  }
}
```

#### PUT /user/profile
**Purpose**: Update user profile information (currently only nickname)

**Request**:
```json
{
  "nickname": "Updated Name"
}
```

**Validation Rules**:
- Nickname cannot be empty
- Maximum 50 characters
- Whitespace trimmed automatically

---

### Folder Management Endpoints

#### GET /folders
**Purpose**: List all folders for authenticated user

**Response Logic**:
- Fetch folders WHERE user_id = current_user.id
- Order by created_at DESC (newest first)
- Include word count for each folder
- Return array with total count

#### POST /folders
**Purpose**: Create new vocabulary folder

**Request**:
```json
{
  "name": "Business English",
  "description": "Vocabulary for professional settings"
}
```

**Validation & Business Logic**:
1. Name required, max 100 characters
2. Description optional, max 500 characters
3. Check for duplicate folder names (per user)
4. Create folder linked to current user
5. Return created folder with word_count = 0

#### GET /folders/{folder_id}
**Purpose**: Get folder details with all words and their learning stats

**Critical Data Processing**:
```python
# For each word in folder:
word_data = {
    "id": word.id,
    "word": word.word,
    "translation": word.translation,
    "example_sentence": word.example_sentence,
    "added_at": word.added_at.isoformat(),
    "stats": {
        "category": word_stats.category if word_stats else "not_known",
        "last_5_results": word_stats.last_5_results if word_stats else [],
        "accuracy": word_stats.accuracy if word_stats else 0
    }
}
```

**Authorization**: Folder must belong to authenticated user

#### PUT /folders/{folder_id}
**Purpose**: Update folder name and/or description

**Business Logic**:
- Partial updates supported (can update just name or description)
- Name uniqueness validation (exclude current folder)
- Authorization check: folder.user_id == current_user.id

#### DELETE /folders/{folder_id}
**Purpose**: Delete folder and all associated data

**Cascade Effects** (due to foreign key constraints):
1. Delete all Words in folder
2. Delete all WordStats for those words
3. Delete all QuizSessions for folder
4. Delete all QuizResults for those sessions
5. Delete Folder record

**Response includes deletion count for user confirmation**

---

### Word Management Endpoints

#### POST /words/{folder_id}
**Purpose**: Manually add single word to folder

**Request**:
```json
{
  "word": "computer",
  "translation": "kompyuter",
  "example_sentence": "I use a computer for work."  // Optional
}
```

**Processing Logic**:
1. Validate folder ownership
2. Check word uniqueness within folder (case-insensitive)
3. Store word in lowercase for consistency
4. Create initial WordStats record for user
5. Return word + stats information

**Critical Business Rule**: Words stored in lowercase, but displayed as entered

#### POST /words/upload-photo
**Purpose**: Extract English words from photo using OCR + AI translation

**Technical Flow**:
```
1. Upload Image → FastAPI
2. Image → Google Vision API (text_detection)
3. Raw Text → Word Extraction Algorithm
4. English Words → OpenAI (individual translation requests)
5. Return: [{"word": "meeting", "translation": "yig'ilish", "confidence": 0.95}]
```

**Word Extraction Algorithm**:
```python
def extract_english_words(text: str) -> List[str]:
    # 1. Regex extraction: \b[a-zA-Z]{3,}\b (letters only, 3+ chars)
    # 2. Convert to lowercase
    # 3. Remove duplicates
    # 4. Filter out common stop words (the, and, for, etc.)
    # 5. Limit to 20 words (prevent overwhelming)
    # 6. Return unique, meaningful words
```

**Performance Considerations**:
- Google Vision API: ~2-3 seconds per image
- OpenAI translation: ~0.5 seconds per word
- Total processing time: ~5-10 seconds for typical image

#### POST /words/{folder_id}/bulk-add
**Purpose**: Add multiple words at once (from OCR results)

**Request**:
```json
{
  "words": [
    {
      "word": "success",
      "translation": "muvaffaqiyat",
      "example_sentence": "Success requires hard work."
    },
    {
      "word": "achievement",
      "translation": "yutuq",
      "example_sentence": null  // Optional
    }
  ]
}
```

**Processing Logic**:
1. Validate each word in batch
2. Skip invalid entries (empty word/translation)
3. Skip duplicates (case-insensitive check)
4. Create Word + WordStats for each valid entry
5. Return count of successfully added words

**Error Handling**: Partial success allowed (some words may fail validation)

#### POST /words/bulk-delete
**Purpose**: Delete multiple words at once

**Request**:
```json
{
  "word_ids": [1, 3, 5, 7, 9]
}
```

**Security Logic**:
1. Validate all word IDs belong to user's folders
2. Only delete words user owns
3. CASCADE DELETE removes WordStats and QuizResults
4. Return confirmation with deleted word details

#### POST /words/generate-example
**Purpose**: AI-generated example sentence for word

**Request**:
```json
{
  "word": "computer",
  "translation": "kompyuter"
}
```

**OpenAI Prompt Engineering**:
```
Create a simple, clear example sentence using the English word "computer" 
(which means "kompyuter" in Uzbek).

Requirements:
- Use everyday, simple English
- Make the sentence practical and useful for language learners
- Keep it under 15 words
- Make sure the meaning of "computer" is clear from context
```

**Response Processing**:
- Clean quotes and extra formatting
- Ensure proper punctuation
- Fallback sentence if AI fails

#### GET /words/{word_id}
**Purpose**: Get detailed word information with learning statistics

**Data Aggregation**:
```json
{
  "word": {
    "id": 1,
    "word": "meeting",
    "translation": "yig'ilish",
    "example_sentence": "We have a meeting at 3 PM.",
    "added_at": "2025-01-15T12:00:00Z",
    "folder": {"id": 1, "name": "Business English"}
  },
  "stats": {
    "category": "normal",
    "last_5_results": [true, false, true, true, false],
    "total_attempts": 15,
    "correct_attempts": 9,
    "accuracy": 60,
    "last_quiz_date": "2025-01-15T14:30:00Z"
  }
}
```

---

### Quiz Engine Endpoints

#### POST /quiz/{folder_id}/start
**Purpose**: Initialize quiz session with first question

**Request**:
```json
{
  "quiz_type": "translation_blitz"  // anagram | translation_blitz | word_blitz | reading
}
```

**Critical Validation Logic**:
```python
# Get complete words (with example sentences)
complete_words = [word for word in folder.words if word.is_complete]

# Check minimum requirements
min_words = settings.min_words_for_quiz  # 5
if quiz_type == "reading":
    min_words = settings.min_words_for_reading  # 8

if len(complete_words) < min_words:
    raise HTTPException(400, f"Need at least {min_words} complete words")
```

**Quiz Session Creation**:
1. Create QuizSession record in database
2. Select and shuffle quiz words (max 10 for regular, 8 for reading)
3. Store session data in memory for real-time processing
4. Generate first question based on quiz type

**Response Structure**:
```json
{
  "session": {
    "session_id": "quiz_789abc",
    "quiz_type": "translation_blitz",
    "folder_id": 1,
    "total_questions": 10
  },
  "question": {
    "question_number": 1,
    "word_id": 5,
    "question": "What is the English word for: 'yig'ilish'?",
    "options": ["meeting", "greeting", "seating"],
    "time_limit": 30
  }
}
```

#### POST /quiz/{session_id}/answer
**Purpose**: Process quiz answer and return next question

**Answer Processing Logic**:
```python
# 1. Validate session exists and belongs to user
# 2. Get current question word data
# 3. Check answer correctness (case-insensitive)
correct_answer = word_data["word"]
is_correct = request.answer.lower().strip() == correct_answer.lower()

# 4. Create QuizResult record
quiz_result = QuizResult(
    session_id=session_id,
    word_id=word_id,
    is_correct=is_correct,
    time_taken=request.time_taken
)

# 5. Update WordStats with new result
word_stats.add_result(is_correct)  # Triggers category update

# 6. Generate next question or signal completion
```

**WordStats Update Algorithm**:
```python
def add_result(self, is_correct: bool):
    # Maintain exactly last 5 results
    results = self.last_5_results or []
    results.append(is_correct)
    if len(results) > 5:
        results = results[-5:]  # Keep only last 5
    self.last_5_results = results
    
    # Update counters
    self.total_attempts += 1
    if is_correct:
        self.correct_attempts += 1
    
    # Auto-update category
    self.update_category()
```

#### POST /quiz/{session_id}/complete
**Purpose**: Finalize quiz session and return comprehensive results

**Results Calculation**:
```python
# Calculate performance metrics
total_questions = len(quiz_data["answers"])
correct_answers = sum(1 for answer in quiz_data["answers"] if answer["is_correct"])
accuracy = round((correct_answers / total_questions) * 100)
total_time = sum(answer["time_taken"] for answer in quiz_data["answers"])

# Identify learning patterns
strongest_words = [word for answer in answers if answer["is_correct"]]
needs_practice = [word for answer in answers if not answer["is_correct"]]

# Get category changes
updated_categories = [
    {
        "word_id": word_id,
        "old_category": "normal",
        "new_category": "strong"  # After quiz performance
    }
]
```

**Session Cleanup**:
1. Update QuizSession.duration in database
2. Remove session from in-memory cache
3. Return comprehensive performance analysis

---

### Voice Conversation Endpoints

#### GET /voice/agents
**Purpose**: List available AI conversation agents

**Response**:
```json
{
  "agents": [
    {
      "id": 1,
      "topic": "cars",
      "title": "Car Expert",
      "description": "Discuss everything about automobiles, engines, and driving",
      "image_url": "https://images.unsplash.com/photo-1550355291-bbee04a92027?w=400",
      "agent_id": "agent_cars_456",
      "is_active": true
    }
  ]
}
```

#### POST /voice/topic/start
**Purpose**: Initiate voice conversation with specific agent

**Processing Flow**:
1. Validate agent exists and is active
2. Generate unique session ID
3. Get WebSocket URL from ElevenLabs service
4. Store session metadata in memory
5. Return connection details with expiration

**ElevenLabs Integration**:
```python
async def get_conversation_url(agent_id: str) -> str:
    # For public agents: Direct WebSocket connection
    websocket_url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={agent_id}"
    
    # For private agents: Request signed URL
    # response = requests.post("/convai/conversations", {"agent_id": agent_id})
    # return response.json()["signed_url"]
```

#### POST /voice/topic/stop
**Purpose**: End voice conversation and calculate session statistics

**Session Management**:
- Validate session belongs to current user
- Calculate duration from start/end timestamps
- Remove session from memory cache
- Return session statistics for analytics

---

## Data Flow Architecture

### 1. User Registration Flow
```
Apple Sign In → Identity Token → Apple Verification → Email Extraction → 
Database Check → User Creation/Update → JWT Generation → Client Response
```

### 2. Word Addition Flow (OCR)
```
Photo Upload → Google Vision OCR → Text Extraction → Word Filtering → 
OpenAI Translation → Word Review → Bulk Add → Database Storage → 
WordStats Creation
```

### 3. Quiz Execution Flow
```
Quiz Start → Folder Validation → Word Selection → Session Creation → 
Question Generation → User Answer → Answer Validation → WordStats Update → 
Category Recalculation → Next Question/Completion
```

### 4. Learning Progress Flow
```
Quiz Answer → WordStats.add_result() → Last 5 Results Update → 
Category Algorithm → Database Update → Progress Tracking
```

---

## Critical Algorithms & Business Logic

### 1. Word Categorization Algorithm

**Core Algorithm**:
```python
def update_category(self):
    if not self.last_5_results or len(self.last_5_results) < 3:
        self.category = "not_known"
        return
    
    # Calculate accuracy from last 5 results
    correct_count = sum(self.last_5_results)
    recent_accuracy = correct_count / len(self.last_5_results)
    
    if recent_accuracy >= 0.8:      # 80% or higher
        self.category = "strong"
    elif recent_accuracy >= 0.5:    # 50-79%
        self.category = "normal"
    else:                          # Below 50%
        self.category = "not_known"
```

**Business Logic Reasoning**:
- **Minimum 3 attempts**: Prevents premature categorization
- **Last 5 results only**: Focuses on recent performance, allows improvement
- **Three tiers**: Simple progression system for user motivation
- **Immediate feedback**: Category updates after each quiz

### 2. Quiz Question Generation

#### Anagram Attack Algorithm:
```python
def generate_anagram_question(word_data):
    # 1. Shuffle letters ensuring different result
    letters = list(word_data["word"].lower())
    shuffled = letters.copy()
    attempts = 0
    while ''.join(shuffled) == word_data["word"] and attempts < 10:
        random.shuffle(shuffled)
        attempts += 1
    
    # 2. Present scrambled letters + Uzbek hint
    return {
        "question": f"Rearrange: {shuffled.upper()}",
        "hint": word_data["translation"],
        "time_limit": 45,
        "correct_answer": word_data["word"]
    }
```

#### Translation Blitz Algorithm:
```python
def generate_translation_question(word_data, all_words):
    # 1. Get wrong options (avoid similar words)
    wrong_options = filter_quiz_options(
        correct_word=word_data["word"], 
        all_words=[w["word"] for w in all_words],
        count=2
    )
    
    # 2. Create multiple choice
    options = [word_data["word"]] + wrong_options
    random.shuffle(options)
    
    return {
        "question": f"What is the English word for: '{word_data['translation']}'?",
        "options": options,
        "time_limit": 30,
        "correct_answer": word_data["word"]
    }
```

### 3. Smart Option Filtering
```python
def filter_quiz_options(correct_word: str, all_words: List[str], count: int = 2):
    wrong_options = []
    
    for word in all_words:
        if word.lower() != correct_word.lower():
            # Avoid too similar words (length, character overlap)
            if not is_similar_word(correct_word, word, threshold=0.8):
                wrong_options.append(word)
            
            if len(wrong_options) >= count:
                break
    
    # Fallback: Use any available words if not enough different ones
    if len(wrong_options) < count:
        remaining = [w for w in all_words if w.lower() != correct_word.lower()]
        random.shuffle(remaining)
        wrong_options.extend(remaining[:count - len(wrong_options)])
    
    return wrong_options[:count]
```

### 4. Reading Comprehension Generation

**OpenAI Prompt Engineering**:
```python
prompt = f"""
Create a short reading passage (3-4 sentences) using these words: {word_list}.

Replace some of the words with blanks like this: _____.
Make the passage natural and interesting.
Return the passage and the correct answers.

Format:
PASSAGE: [passage with blanks]
ANSWERS: [list of correct words for each blank]
"""
```

**Processing Algorithm**:
1. Select 8 best words from folder (with examples)
2. Send to OpenAI for passage generation
3. Parse response into passage + answer key
4. Present as fill-in-the-blank exercise
5. Validate user answers against key

---

## External Service Integrations

### 1. Apple Sign In Integration

**Token Verification Process**:
```python
async def verify_identity_token(identity_token: str):
    # 1. Decode JWT header to get key ID (kid)
    unverified_header = jwt.get_unverified_header(identity_token)
    kid = unverified_header.get("kid")
    
    # 2. Fetch Apple's public keys
    keys_response = requests.get("https://appleid.apple.com/auth/keys")
    apple_keys = keys_response.json()
    
    # 3. Find matching public key
    public_key_data = get_key_by_kid(kid, apple_keys)
    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(public_key_data)
    
    # 4. Verify and decode token
    decoded_token = jwt.decode(
        identity_token,
        public_key,
        algorithms=["RS256"],
        audience=settings.apple_team_id,
        issuer="https://appleid.apple.com"
    )
    
    # 5. Extract user data
    return {
        "email": decoded_token.get("email"),
        "email_verified": decoded_token.get("email_verified"),
        "sub": decoded_token.get("sub")
    }
```

**Security Features**:
- RSA256 signature verification
- Audience validation (app-specific)
- Issuer validation (Apple)
- Token expiration checking
- Public key rotation support

### 2. Google Vision API Integration

**OCR Processing Pipeline**:
```python
async def extract_text_from_image(image_file: UploadFile):
    # 1. Read image content
    content = await image_file.read()
    
    # 2. Call Google Vision API
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    payload = {
        "requests": [{
            "image": {"content": base64.b64encode(content).decode()},
            "features": [{"type": "TEXT_DETECTION", "maxResults": 50}]
        }]
    }
    
    # 3. Process response
    response = requests.post(url, json=payload)
    text_annotations = response.json()["responses"][0]["textAnnotations"]
    
    # 4. Extract and filter English words
    full_text = text_annotations[0]["description"]
    english_words = extract_english_words(full_text)
    
    return english_words
```

**Word Extraction Logic**:
```python
def extract_english_words(text: str) -> List[str]:
    # 1. Regex: Extract letter-only words (3+ chars)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    
    # 2. Normalize: lowercase, remove duplicates
    unique_words = list(set([word.lower() for word in words]))
    
    # 3. Filter: Remove common stop words
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', ...}
    filtered_words = [w for w in unique_words if w not in stop_words and len(w) >= 4]
    
    # 4. Limit: Max 20 words to prevent overwhelming
    return filtered_words[:20]
```

### 3. OpenAI Integration

**Translation Service**:
```python
async def translate_to_uzbek(english_word: str) -> str:
    prompt = f"""
    Translate this English word to Uzbek. Return only the Uzbek translation.
    
    English word: {english_word}
    Uzbek translation:
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Professional English-Uzbek translator."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50,
        temperature=0.1  # Low temperature for consistency
    )
    
    translation = response.choices[0].message.content.strip()
    return clean_translation_response(translation)
```

**Example Generation Service**:
```python
async def generate_example_sentence(english_word: str, uzbek_translation: str) -> str:
    prompt = f"""
    Create a simple example sentence using "{english_word}" (means "{uzbek_translation}" in Uzbek).
    
    Requirements:
    - Simple, everyday English
    - Under 15 words
    - Clear context for word meaning
    - Practical for language learners
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "English teacher creating vocabulary examples."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.3  # Some creativity for varied examples
    )
    
    return clean_example_response(response.choices[0].message.content)
```

### 4. ElevenLabs Voice Integration

**WebSocket Connection Management**:
```python
async def get_conversation_url(agent_id: str) -> str:
    # Direct WebSocket for public agents
    if is_public_agent(agent_id):
        return f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={agent_id}"
    
    # Signed URL for private agents
    url = "https://api.elevenlabs.io/v1/convai/conversations"
    headers = {"xi-api-key": settings.elevenlabs_api_key}
    payload = {"agent_id": agent_id}
    
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["signed_url"]
```

**Voice Session Management**:
```python
# In-memory session storage
active_voice_sessions = {
    "session_id": {
        "user_id": 1,
        "agent_id": "agent_cars_456",
        "topic": "cars",
        "started_at": datetime.utcnow(),
        "websocket_url": "wss://..."
    }
}

# Session cleanup (30-minute timeout)
def cleanup_expired_sessions():
    current_time = datetime.utcnow()
    expired = [
        sid for sid, data in active_voice_sessions.items()
        if current_time - data["started_at"] > timedelta(minutes=30)
    ]
    for sid in expired:
        del active_voice_sessions[sid]
```

---

## Authentication & Security

### JWT Token System

**Token Generation**:
```python
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.jwt_expire_minutes)  # 30 days
    to_encode.update({"exp": expire})
    
    return jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
```

**Token Verification Middleware**:
```python
def verify_token(credentials: HTTPAuthorizationCredentials):
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(401, "Invalid token")
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token")
```

### Data Security Measures

**Input Validation**:
- All user inputs sanitized and validated
- SQL injection prevention via SQLAlchemy ORM
- XSS prevention through proper encoding
- File upload validation (image types only)

**Authorization Checks**:
```python
# Example: Folder access control
folder = db.query(Folder).filter(
    Folder.id == folder_id,
    Folder.user_id == current_user.id  # User isolation
).first()

if not folder:
    raise HTTPException(404, "Folder not found")
```

**API Key Management**:
- External API keys stored in environment variables
- No API keys exposed in client-side code
- Service-specific error handling for API failures

---

## Quiz Engine Deep Dive

### Quiz Type Implementations

#### 1. Anagram Attack
**Concept**: Unscramble English letters using Uzbek translation as hint
**Algorithm**:
```python
def generate_anagram(word: str) -> str:
    letters = list(word.lower())
    attempts = 0
    while attempts < 10:
        random.shuffle(letters)
        shuffled = ''.join(letters)
        if shuffled != word.lower():  # Ensure it's actually scrambled
            return shuffled.upper()
        attempts += 1
    return word.upper()  # Fallback
```

**UI Flow**:
1. Display scrambled letters: "ETNGMIE"
2. Show hint: "yig'ilish"
3. User reconstructs: "MEETING"
4. Validation: Case-insensitive exact match
5. Time limit: 45 seconds

#### 2. Translation Blitz
**Concept**: See Uzbek translation, choose correct English word
**Algorithm**:
```python
def generate_translation_options(correct_word: str, all_words: List[str]) -> List[str]:
    # 1. Filter out similar words to avoid confusion
    candidates = [w for w in all_words if not is_similar_word(correct_word, w)]
    
    # 2. Select 2 random wrong options
    wrong_options = random.sample(candidates, min(2, len(candidates)))
    
    # 3. Add correct answer and shuffle
    options = [correct_word] + wrong_options
    random.shuffle(options)
    
    return options
```

**Similarity Detection**:
```python
def is_similar_word(word1: str, word2: str, threshold: float = 0.8) -> bool:
    word1, word2 = word1.lower(), word2.lower()
    
    # Length similarity
    if abs(len(word1) - len(word2)) <= 1:
        # Character overlap check
        common_chars = set(word1) & set(word2)
        overlap_ratio = len(common_chars) / max(len(set(word1)), len(set(word2)))
        return overlap_ratio > threshold
    
    return False
```

#### 3. Word Blitz
**Concept**: See example sentence, choose word that fits
**Algorithm**: Similar to Translation Blitz but uses example sentences
**Challenge**: Requires meaningful example sentences (why incomplete words are excluded)

#### 4. Reading Comprehension
**Concept**: AI-generated passage with word gaps to fill
**Implementation**:
```python
def generate_reading_passage(words: List[Dict]) -> Dict:
    # 1. Select 8 words maximum for manageable passage
    selected_words = words[:8]
    word_list = [w["word"] for w in selected_words]
    
    # 2. Generate passage via OpenAI
    prompt = f"""
    Create a 3-4 sentence passage using these words: {', '.join(word_list)}
    Replace some words with _____ blanks.
    Make it natural and interesting.
    
    Format:
    PASSAGE: [text with blanks]
    ANSWERS: [word1, word2, word3]
    """
    
    # 3. Parse AI response
    response = await openai_service.generate_quiz_question(selected_words, "reading")
    passage, answers = parse_reading_response(response)
    
    return {
        "passage": passage,
        "correct_answers": answers,
        "time_limit": 120  # 2 minutes for reading
    }
```

### Quiz Session State Management

**In-Memory Session Storage**:
```python
quiz_sessions = {
    "quiz_session_123": {
        "session_id": 123,
        "folder_id": 1,
        "quiz_type": "translation_blitz",
        "words": [
            {"id": 1, "word": "meeting", "translation": "yig'ilish"},
            {"id": 2, "word": "computer", "translation": "kompyuter"}
        ],
        "current_question": 0,
        "answers": [
            {"word_id": 1, "answer": "meeting", "is_correct": True, "time_taken": 15}
        ]
    }
}
```

**Session Lifecycle**:
1. **Creation**: Quiz start → Database QuizSession + Memory session
2. **Active**: Answer submission → Update memory + Database QuizResult
3. **Completion**: Final results → Calculate stats + Cleanup memory
4. **Timeout**: Auto-cleanup after 30 minutes of inactivity

### Performance Tracking Algorithm

**Real-time Category Updates**:
```python
def process_quiz_answer(word_id: int, user_id: int, is_correct: bool):
    # 1. Get or create WordStats
    stats = get_word_stats(word_id, user_id)
    
    # 2. Add result and trigger category update
    stats.add_result(is_correct)
    
    # 3. Category algorithm
    if len(stats.last_5_results) < 3:
        stats.category = "not_known"
    else:
        accuracy = sum(stats.last_5_results) / len(stats.last_5_results)
        if accuracy >= 0.8:
            stats.category = "strong"
        elif accuracy >= 0.5:
            stats.category = "normal"
        else:
            stats.category = "not_known"
    
    # 4. Database update
    db.commit()
    
    return stats.category
```

---

## Voice Conversation System

### Architecture Overview
```
User Voice Input → WebSocket → ElevenLabs Agent → 
AI Processing (GPT + TTS) → Voice Response → User
```

### Agent Configuration

**Topic-Specific Prompts** (Managed in ElevenLabs dashboard):

**Cars Agent**:
```
You are a passionate car expert. You ONLY discuss automotive topics:
- Car models, specifications, and reviews
- Engine types and performance
- Driving techniques and safety
- Car maintenance and repairs
- Automotive industry news

Use simple, engaging language. If asked about non-car topics, redirect to automobiles.
```

**Football Agent**:
```
You are an enthusiastic football (soccer) coach. You ONLY discuss:
- Team tactics and formations
- Player analysis and transfers
- Match predictions and reviews
- Training techniques
- Football history and statistics

Keep conversations exciting and knowledgeable. Redirect non-football topics back to soccer.
```

**Travel Agent**:
```
You are an experienced travel guide. Focus ONLY on:
- Destination recommendations
- Cultural insights and customs
- Travel tips and planning
- Transportation and accommodation
- Local cuisine and attractions

Be helpful and inspiring about travel. Redirect other topics to travel discussions.
```

### Technical Implementation

**Session Management**:
```python
class VoiceSessionManager:
    def __init__(self):
        self.active_sessions = {}
    
    def create_session(self, user_id: int, agent_id: str) -> str:
        session_id = f"voice_{uuid.uuid4().hex[:12]}"
        
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "agent_id": agent_id,
            "started_at": datetime.utcnow(),
            "websocket_url": None,
            "status": "initializing"
        }
        
        return session_id
    
    def cleanup_expired(self):
        # Remove sessions older than 30 minutes
        cutoff = datetime.utcnow() - timedelta(minutes=30)
        expired = [
            sid for sid, data in self.active_sessions.items()
            if data["started_at"] < cutoff
        ]
        for sid in expired:
            del self.active_sessions[sid]
```

**ElevenLabs Integration**:
```python
async def establish_voice_connection(agent_id: str) -> str:
    # 1. Generate WebSocket URL
    if is_public_agent(agent_id):
        # Direct connection for public agents
        websocket_url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={agent_id}"
    else:
        # Signed URL for private agents
        signed_url = await request_signed_url(agent_id)
        websocket_url = signed_url
    
    # 2. Return connection details
    return {
        "websocket_url": websocket_url,
        "connection_expires_at": datetime.utcnow() + timedelta(minutes=30)
    }
```

### WebSocket Communication Protocol

**Client → Server Events**:
```javascript
// Send audio data
{
  "user_audio_chunk": "base64_encoded_audio_data"
}

// Send text message
{
  "type": "user_message", 
  "text": "Tell me about electric cars"
}

// Handle interruption
{
  "type": "interruption"
}
```

**Server → Client Events**:
```javascript
// Agent audio response
{
  "type": "audio",
  "audio_event": {
    "audio_base_64": "base64_encoded_response",
    "event_id": 1
  }
}

// Agent text response
{
  "type": "agent_response",
  "agent_response_event": {
    "agent_response": "Electric cars are revolutionizing transportation..."
  }
}

// User speech transcript
{
  "type": "user_transcript",
  "user_transcription_event": {
    "user_transcript": "What about Tesla Model S?"
  }
}
```

### Interruption Handling

**ElevenLabs Built-in Features**:
- **Voice Activity Detection**: Automatically detects when user starts speaking
- **Interrupt Handling**: Stops agent speech when user interrupts
- **Turn-taking Model**: Understands conversation flow and pauses
- **Real-time Processing**: Low-latency audio streaming

**Configuration Options**:
```python
# In ElevenLabs agent settings
interruption_settings = {
    "interruptions_enabled": True,
    "turn_timeout": 3,  # Seconds to wait for user input
    "sensitivity": "high"  # Interruption detection sensitivity
}
```

---

## Performance & Scalability

### Database Optimization

**Indexing Strategy**:
```sql
-- User lookup optimization
CREATE INDEX idx_users_email ON users(email);

-- Folder access optimization  
CREATE INDEX idx_folders_user_id ON folders(user_id);
CREATE INDEX idx_folders_user_created ON folders(user_id, created_at DESC);

-- Word lookup optimization
CREATE INDEX idx_words_folder_id ON words(folder_id);
CREATE INDEX idx_words_folder_word ON words(folder_id, word);

-- WordStats lookup optimization
CREATE INDEX idx_word_stats_user_word ON word_stats(user_id, word_id);
CREATE INDEX idx_word_stats_word_id ON word_stats(word_id);

-- Quiz performance optimization
CREATE INDEX idx_quiz_sessions_user ON quiz_sessions(user_id, created_at DESC);
CREATE INDEX idx_quiz_results_session ON quiz_results(session_id);
```

**Query Optimization Examples**:
```python
# Efficient folder loading with word counts
folders = db.query(Folder).filter(
    Folder.user_id == user_id
).options(
    selectinload(Folder.words)  # Eager load to avoid N+1 queries
).order_by(Folder.created_at.desc()).all()

# Efficient word stats loading
word_stats = db.query(WordStats).filter(
    WordStats.user_id == user_id,
    WordStats.word_id.in_(word_ids)
).all()
```

### Caching Strategy

**In-Memory Caching**:
```python
# Quiz sessions (temporary, 30-minute TTL)
quiz_sessions_cache = {}

# Voice sessions (temporary, 30-minute TTL)  
voice_sessions_cache = {}

# Apple public keys (1-hour TTL)
apple_keys_cache = {
    "keys": None,
    "expires_at": None
}
```

**Redis Integration** (Future Enhancement):
```python
import redis

redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=0,
    decode_responses=True
)

# Cache user profiles
def cache_user_profile(user_id: int, profile_data: dict):
    redis_client.setex(
        f"user_profile:{user_id}",
        3600,  # 1 hour TTL
        json.dumps(profile_data)
    )
```

### API Rate Limiting

**External Service Limits**:
- **Google Vision API**: 1,800 requests/minute
- **OpenAI API**: 10,000 requests/minute (GPT-4o-mini)
- **ElevenLabs**: Based on subscription tier

**Rate Limiting Implementation**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/words/upload-photo")
@limiter.limit("10/minute")  # Limit OCR requests
async def upload_photo_ocr(request: Request, ...):
    # OCR processing
    pass

@app.post("/words/generate-example")  
@limiter.limit("30/minute")  # Limit AI generation
async def generate_example(request: Request, ...):
    # Example generation
    pass
```

### Horizontal Scaling Considerations

**Stateless Design**:
- JWT tokens (no server-side sessions)
- Database-persisted user state
- In-memory caches can be moved to Redis

**Database Scaling**:
```python
# Read replicas for analytics queries
ANALYTICS_DATABASE_URL = "postgresql://readonly_user:password@replica_host/db"

# Separate engines for read/write operations
write_engine = create_engine(settings.database_url)
read_engine = create_engine(settings.analytics_database_url)
```

**Microservices Architecture** (Future):
```
API Gateway → 
├── User Service (Authentication, Profiles)
├── Vocabulary Service (Folders, Words, Stats) 
├── Quiz Service (Quiz Engine, Results)
├── Voice Service (ElevenLabs Integration)
└── AI Service (OpenAI, Google Vision)
```

### Monitoring & Analytics

**Application Metrics**:
```python
import logging
from prometheus_client import Counter, Histogram

# Custom metrics
quiz_attempts = Counter('quiz_attempts_total', 'Total quiz attempts', ['quiz_type'])
api_request_duration = Histogram('api_request_duration_seconds', 'API request duration')

# Usage in endpoints
@quiz_router.post("/{folder_id}/start")
async def start_quiz(...):
    quiz_attempts.labels(quiz_type=request.quiz_type).inc()
    with api_request_duration.time():
        # Quiz logic
        pass
```

**Error Tracking**:
```python
import sentry_sdk

sentry_sdk.init(
    dsn=settings.sentry_dsn,
    traces_sample_rate=1.0,
)

# Automatic error capture with context
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    sentry_sdk.capture_exception(exc)
    # Error response
```

---

## Setup & Deployment

### Development Setup

**1. Clone and Environment Setup**:
```bash
git clone <repository-url>
cd vocabulary-app

python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

**2. Environment Configuration**:
```env

```

**3. Database Initialization**:
```bash
# Create tables
python -c "from app.database import create_tables; create_tables()"

# Initialize voice agents
python scripts/init_voice_agents.py create

# Verify setup
python scripts/init_voice_agents.py list
```

**4. Run Development Server**:
```bash
uvicorn app.main:app --reload --port 8000

# API Documentation: http://localhost:8000/docs
# Health Check: http://localhost:8000/health
```

### Production Deployment (Railway)

**1. Railway Setup**:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway add postgresql
railway deploy
```

**2. Environment Variables** (Railway Dashboard):
```
DATABASE_URL=<railway_postgresql_url>
GOOGLE_VISION_API_KEY=<your_key>
ELEVENLABS_API_KEY=<your_key>
OPENAI_API_KEY=<your_key>
APPLE_TEAM_ID=<your_team_id>
APPLE_KEY_ID=<your_key_id>
JWT_SECRET_KEY=<secure_random_key>
DEBUG=False
PORT=8000
```

**3. Production Configuration**:
```python
# app/config.py - Production overrides
class ProductionSettings(Settings):
    debug: bool = False
    
    # Database connection pooling
    database_pool_size: int = 20
    database_max_overflow: int = 30
    
    # Security
    jwt_expire_minutes: int = 43200  # 30 days
    
    # Rate limiting
    enable_rate_limiting: bool = True
    
    # CORS configuration
    allowed_origins: List[str] = [
        "https://yourdomain.com",
        "https://app.yourdomain.com"
    ]
```

**4. Health Checks & Monitoring**:
```python
@app.get("/health")
async def health_check():
    # Database connectivity check
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    finally:
        db.close()
    
    # External services check
    services = {
        "database": db_status,
        "openai": await check_openai_health(),
        "google_vision": await check_vision_health(),
        "elevenlabs": await check_elevenlabs_health()
    }
    
    return {
        "status": "healthy" if all(s == "healthy" for s in services.values()) else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "services": services,
        "version": "1.0.0"
    }
```

### Mobile App Integration

**API Client Setup**:
```swift
// iOS Swift example
class VocabularyAPI {
    private let baseURL = "https://your-api.railway.app"
    private var accessToken: String?
    
    func authenticateWithApple(identityToken: String) async -> AuthResponse? {
        let request = AppleSignInRequest(
            identityToken: identityToken,
            userIdentifier: userIdentifier,
            nickname: nickname
        )
        
        // POST /auth/apple-signin
        let response = await post("/auth/apple-signin", body: request)
        self.accessToken = response?.accessToken
        return response
    }
    
    func getFolders() async -> [Folder] {
        // GET /folders with Authorization header
        return await get("/folders", headers: authHeaders)
    }
}
```

**WebSocket Integration**:
```swift
// Voice conversation WebSocket
class VoiceConversationManager: ObservableObject {
    private var webSocket: URLSessionWebSocketTask?
    
    func startConversation(withAgent agentId: Int) async {
        // 1. Get WebSocket URL from API
        let session = await api.startVoiceSession(agentId: agentId)
        
        // 2. Connect to ElevenLabs WebSocket
        guard let url = URL(string: session.websocketUrl) else { return }
        webSocket = URLSession.shared.webSocketTask(with: url)
        webSocket?.resume()
        
        // 3. Handle messages
        receiveMessages()
    }
    
    private func receiveMessages() {
        webSocket?.receive { [weak self] result in
            switch result {
            case .success(let message):
                self?.handleWebSocketMessage(message)
                self?.receiveMessages() // Continue listening
            case .failure(let error):
                print("WebSocket error: \(error)")
            }
        }
    }
}
```

---

This comprehensive technical documentation provides a complete understanding of the Vocabulary Learning App's architecture, algorithms, and implementation details. The system is designed for scalability, maintainability, and optimal user experience through intelligent learning algorithms and modern AI integrations.