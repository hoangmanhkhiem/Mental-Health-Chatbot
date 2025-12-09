"""
AWE Mental Health Chatbot - Enhanced FastAPI Server
====================================================

Server với Enhanced Therapeutic Chatbot bao gồm 8 layers nâng cao:
1. Conversational Layer
2. Personalization Layer  
3. Emotional Understanding Layer
4. Storytelling Therapy Mode
5. RAG Precision Boost
6. Reasoning Layer
7. Safety & Ethics Layer
8. Proactive Dialogue Engine

Sử dụng thay cho server.py khi muốn dùng tất cả tính năng nâng cao.
"""

import os
import logging
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Twilio for WhatsApp
from twilio.rest import Client
from twilio.request_validator import RequestValidator

# Database setup
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from database_aad import get_database_engine
from database import Base, set_engine_and_session

# Enhanced Chatbot + RAG
from enhanced_chatbot import EnhancedTherapeuticChatbot, create_enhanced_chatbot
from rag_system_v2 import TherapeuticRAG

# Layers for additional endpoints
from layers import UserMemoryStore, ProactiveDialogueEngine

# --------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# DATABASE SETUP
# --------------------------------------------------------------
engine = get_database_engine()
session_factory = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)

set_engine_and_session(engine, session_factory)
SessionLocal = session_factory

# --------------------------------------------------------------
# TWILIO SETUP
# --------------------------------------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

twilio_client = None
twilio_validator = None

if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    twilio_validator = RequestValidator(TWILIO_AUTH_TOKEN)
    logger.info(f"✓ Twilio client initialized with number: {TWILIO_WHATSAPP_NUMBER}")
else:
    logger.warning("⚠️ Twilio credentials not set - WhatsApp responses will not be sent")

# --------------------------------------------------------------
# APPLICATION LIFESPAN: STARTUP & SHUTDOWN
# --------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI startup and shutdown lifecycle"""

    # ---------- STARTUP ----------
    logger.info("🚀 Starting Enhanced Chatbot initialization...")

    # Database test
    try:
        logger.info("🔐 Testing database connection...")
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        logger.info("✓ Database connection verified")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {e}")
        raise

    # RAG Initialization
    rag_system = None
    try:
        logger.info("📚 Initializing knowledge base (RAG system)...")
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if not google_api_key:
            logger.warning("⚠️ GOOGLE_API_KEY not set - RAG system will not be initialized")
        else:
            rag_system = TherapeuticRAG(google_api_key=google_api_key)
            logger.info("✓ Knowledge base initialized with Gemini embeddings")

    except Exception as e:
        logger.warning(f"⚠️ RAG system initialization failed: {e}")
        rag_system = None

    # Enhanced Chatbot Initialization
    try:
        logger.info("🤖 Initializing Enhanced Therapeutic Chatbot with 8 layers...")
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if not google_api_key:
            logger.error("✗ GOOGLE_API_KEY not set!")
            raise ValueError("GOOGLE_API_KEY is required")

        chatbot = create_enhanced_chatbot(
            google_api_key=google_api_key,
            rag_system=rag_system,
            db_session_factory=session_factory
        )

        app.state.chatbot = chatbot
        app.state.rag_system = rag_system

        logger.info("✓ Enhanced Chatbot initialized successfully with all layers:")
        logger.info("  ├─ Layer 1: Conversational Layer ✓")
        logger.info("  ├─ Layer 2: Personalization Layer ✓")
        logger.info("  ├─ Layer 3: Emotional Understanding ✓")
        logger.info("  ├─ Layer 4: Storytelling Therapy ✓")
        logger.info("  ├─ Layer 5: RAG Precision Boost ✓")
        logger.info("  ├─ Layer 6: Reasoning Layer ✓")
        logger.info("  ├─ Layer 7: Safety & Ethics ✓")
        logger.info("  └─ Layer 8: Proactive Dialogue ✓")

    except Exception as e:
        logger.error(f"✗ Enhanced Chatbot initialization failed: {e}")
        raise

    # Startup banner
    logger.info("""
╔══════════════════════════════════════════════════════════════════╗
║   AWE Enhanced Therapeutic Chatbot - Multi-Channel Production    ║
║        WhatsApp + Web Chat + Telegram | Digital Wellness         ║
║                                                                  ║
║   🧠 8 Enhancement Layers Active:                                ║
║      • Emotional Understanding    • Personalization              ║
║      • Conversational Naturalize  • Storytelling Therapy         ║
║      • RAG Precision Boost        • Reasoning Layer              ║
║      • Safety & Ethics            • Proactive Dialogue           ║
║                                                                  ║
║                  Powered by Google Gemini AI                     ║
╚══════════════════════════════════════════════════════════════════╝
""")

    logger.info("🎉 Enhanced Chatbot startup complete!")
    logger.info("📱 WhatsApp channel: ACTIVE")
    logger.info("💻 Web chat channel: ACTIVE")

    yield

    # ---------- SHUTDOWN ----------
    logger.info("👋 Shutting down enhanced chatbot gracefully...")

    try:
        engine.dispose()
        logger.info("✓ Database connections closed")
    except Exception as e:
        logger.error(f"✗ Shutdown error: {e}")

    logger.info("👋 Shutdown complete")

# --------------------------------------------------------------
# FASTAPI INSTANCE
# --------------------------------------------------------------
app = FastAPI(
    title="AWE Enhanced Mental Health Chatbot",
    description="Therapeutic chatbot with 8 enhancement layers for digital wellness",
    version="2.0.0",
    lifespan=lifespan
)

# --------------------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------------------
cors_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://awedigitalwellness.com",
    "https://www.awedigitalwellness.com",
    "https://*.vercel.app",
]

env_origins = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else []
if env_origins:
    cors_origins.extend(env_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------
# BASIC ROUTES
# --------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AWE Enhanced Therapeutic Chatbot",
        "channels": ["whatsapp", "web"],
        "version": "2.0.0",
        "enhancement_layers": 8,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/health")
async def health():
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))

        return {
            "status": "healthy",
            "service": "AWE Enhanced Therapeutic Chatbot",
            "channels": {
                "whatsapp": "active" if twilio_client else "inactive",
                "web": "active",
            },
            "enhancement_layers": {
                "emotional_understanding": "active",
                "personalization": "active",
                "conversational": "active",
                "storytelling": "active",
                "rag_precision": "active",
                "reasoning": "active",
                "safety_ethics": "active",
                "proactive_dialogue": "active"
            },
            "version": "2.0.0",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")


# --------------------------------------------------------------
# ENHANCED WEB CHAT ENDPOINT
# --------------------------------------------------------------
@app.post("/api/enhanced-chat")
async def enhanced_web_chat(request: Request):
    """
    Enhanced web chat với tất cả 8 layers.
    
    Request body:
    {
        "content": "tin nhắn",
        "conversation_id": "uuid (optional)",
        "use_storytelling": false,
        "use_proactive": true
    }
    """
    try:
        body = await request.json()
        user_message = body.get("content", "").strip()
        conversation_id = body.get("conversation_id")
        use_storytelling = body.get("use_storytelling", False)
        use_proactive = body.get("use_proactive", True)

        logger.info(f"💻 Enhanced chat - Conv: {conversation_id}, Content: {user_message[:50]}...")

        if not user_message:
            return JSONResponse({
                "conversation_id": conversation_id,
                "content": "Vui lòng nhập tin nhắn.",
                "error": True
            }, status_code=400)

        db = SessionLocal()

        try:
            chatbot: EnhancedTherapeuticChatbot = app.state.chatbot

            # Create new conversation if needed
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                logger.info(f"✓ New enhanced conversation: {conversation_id}")

            user_id = f"web:{conversation_id}"

            # Generate enhanced response
            response = chatbot.generate_response(
                db=db,
                user_id=user_id,
                user_message=user_message,
                use_rag=True,
                use_storytelling=use_storytelling,
                use_proactive=use_proactive
            )

            logger.info(f"✓ Enhanced response - Emotion: {response.emotion_detected}, Crisis: {response.is_crisis}")

            return JSONResponse({
                "conversation_id": conversation_id,
                "content": response.response,
                "is_crisis": response.is_crisis,
                "metadata": {
                    "emotion_detected": response.emotion_detected,
                    "emotion_intensity": response.emotion_intensity,
                    "used_rag": response.used_rag,
                    "used_storytelling": response.used_storytelling,
                    "proactive_elements": response.proactive_elements
                }
            })

        except Exception as e:
            logger.error(f"✗ Enhanced chat error: {e}", exc_info=True)
            return JSONResponse({
                "conversation_id": conversation_id,
                "content": "Mình xin lỗi, có lỗi xảy ra. Bạn có thể thử lại không?",
                "error": True
            }, status_code=500)

        finally:
            db.close()

    except Exception as e:
        logger.error(f"✗ Enhanced chat endpoint error: {e}", exc_info=True)
        return JSONResponse({
            "content": "Xin lỗi, có lỗi xảy ra.",
            "error": True
        }, status_code=500)


# --------------------------------------------------------------
# STORYTELLING ENDPOINT
# --------------------------------------------------------------
@app.post("/api/storytelling")
async def storytelling_therapy(request: Request):
    """
    Endpoint cho storytelling therapy mode.
    
    Request body:
    {
        "content": "vấn đề của người dùng",
        "conversation_id": "uuid"
    }
    """
    try:
        body = await request.json()
        user_message = body.get("content", "").strip()
        conversation_id = body.get("conversation_id", str(uuid.uuid4()))

        if not user_message:
            return JSONResponse({
                "content": "Vui lòng chia sẻ vấn đề của bạn.",
                "error": True
            }, status_code=400)

        db = SessionLocal()

        try:
            chatbot: EnhancedTherapeuticChatbot = app.state.chatbot
            user_id = f"web:{conversation_id}"

            # Generate response with storytelling mode
            response = chatbot.generate_response(
                db=db,
                user_id=user_id,
                user_message=user_message,
                use_rag=True,
                use_storytelling=True,
                use_proactive=False
            )

            return JSONResponse({
                "conversation_id": conversation_id,
                "content": response.response,
                "type": "storytelling",
                "metadata": {
                    "emotion_detected": response.emotion_detected,
                    "used_storytelling": response.used_storytelling
                }
            })

        finally:
            db.close()

    except Exception as e:
        logger.error(f"✗ Storytelling endpoint error: {e}")
        return JSONResponse({
            "content": "Xin lỗi, có lỗi xảy ra.",
            "error": True
        }, status_code=500)


# --------------------------------------------------------------
# THERAPEUTIC EXERCISE ENDPOINT
# --------------------------------------------------------------
@app.get("/api/exercise/{conversation_id}")
async def get_therapeutic_exercise(conversation_id: str, issue: str = None):
    """
    Lấy bài tập trị liệu phù hợp.
    
    Query params:
    - issue: Vấn đề cụ thể (optional)
    """
    try:
        chatbot: EnhancedTherapeuticChatbot = app.state.chatbot
        user_id = f"web:{conversation_id}"

        exercise = chatbot.get_therapeutic_exercise(user_id, issue)

        return JSONResponse({
            "conversation_id": conversation_id,
            "exercise": exercise,
            "type": "therapeutic_exercise"
        })

    except Exception as e:
        logger.error(f"✗ Exercise endpoint error: {e}")
        return JSONResponse({
            "error": True,
            "message": str(e)
        }, status_code=500)


# --------------------------------------------------------------
# CONVERSATION SUMMARY ENDPOINT
# --------------------------------------------------------------
@app.get("/api/summary/{conversation_id}")
async def get_conversation_summary(conversation_id: str):
    """
    Lấy tóm tắt cuộc hội thoại.
    """
    try:
        chatbot: EnhancedTherapeuticChatbot = app.state.chatbot
        user_id = f"web:{conversation_id}"

        summary = chatbot.get_conversation_summary(user_id)

        return JSONResponse({
            "conversation_id": conversation_id,
            "summary": summary
        })

    except Exception as e:
        logger.error(f"✗ Summary endpoint error: {e}")
        return JSONResponse({
            "error": True,
            "message": str(e)
        }, status_code=500)


# --------------------------------------------------------------
# CHECK-IN ENDPOINT
# --------------------------------------------------------------
@app.get("/api/check-in/{conversation_id}")
async def get_check_in_message(conversation_id: str):
    """
    Lấy tin nhắn check-in nếu cần.
    """
    try:
        chatbot: EnhancedTherapeuticChatbot = app.state.chatbot
        user_id = f"web:{conversation_id}"

        check_in = chatbot.get_check_in_message(user_id)

        if check_in:
            return JSONResponse({
                "conversation_id": conversation_id,
                "check_in_message": check_in,
                "should_check_in": True
            })
        else:
            return JSONResponse({
                "conversation_id": conversation_id,
                "should_check_in": False
            })

    except Exception as e:
        logger.error(f"✗ Check-in endpoint error: {e}")
        return JSONResponse({
            "error": True,
            "message": str(e)
        }, status_code=500)


# --------------------------------------------------------------
# GREETING ENDPOINT
# --------------------------------------------------------------
@app.get("/api/greeting/{conversation_id}")
async def get_greeting(conversation_id: str):
    """
    Lấy lời chào phù hợp cho người dùng.
    """
    try:
        chatbot: EnhancedTherapeuticChatbot = app.state.chatbot
        user_id = f"web:{conversation_id}"

        greeting = chatbot.generate_greeting(user_id)

        return JSONResponse({
            "conversation_id": conversation_id,
            "greeting": greeting
        })

    except Exception as e:
        logger.error(f"✗ Greeting endpoint error: {e}")
        return JSONResponse({
            "error": True,
            "message": str(e)
        }, status_code=500)


# --------------------------------------------------------------
# BACKWARD COMPATIBLE ENDPOINTS
# --------------------------------------------------------------
@app.post("/api/webChat")
async def web_chat_tracked(request: Request):
    """Backward compatible web chat endpoint."""
    try:
        body = await request.json()
        user_message = body.get("content", "").strip()
        conversation_id = body.get("conversation_id")
        message_index = body.get("message_index", 0)

        if not user_message:
            return JSONResponse({
                "conversation_id": conversation_id,
                "message_index": message_index + 1,
                "content": "Vui lòng nhập tin nhắn.",
                "error": True
            }, status_code=400)

        db = SessionLocal()

        try:
            chatbot: EnhancedTherapeuticChatbot = app.state.chatbot

            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            user_id = f"web:{conversation_id}"

            response = chatbot.generate_response(
                db=db,
                user_id=user_id,
                user_message=user_message,
                use_rag=True,
                use_storytelling=False,
                use_proactive=True
            )

            return JSONResponse({
                "conversation_id": conversation_id,
                "message_index": message_index + 1,
                "content": response.response,
                "is_crisis": response.is_crisis
            })

        finally:
            db.close()

    except Exception as e:
        logger.error(f"✗ webChat error: {e}")
        return JSONResponse({
            "content": "Xin lỗi, có lỗi xảy ra.",
            "error": True
        }, status_code=500)


@app.post("/api/awe-chat")
async def awe_chat(request: Request):
    """Simple backward compatible chat endpoint."""
    try:
        body = await request.json()
        user_message = body.get("message", "").strip()
        
        if not user_message:
            return JSONResponse({
                "response": "Vui lòng nhập tin nhắn.",
                "error": True
            }, status_code=400)

        db = SessionLocal()
        
        try:
            chatbot: EnhancedTherapeuticChatbot = app.state.chatbot
            user_id = f"web-{datetime.utcnow().timestamp()}"

            response = chatbot.generate_response(
                db=db,
                user_id=user_id,
                user_message=user_message,
                use_rag=True,
                use_storytelling=False,
                use_proactive=False
            )

            return JSONResponse({
                "response": response.response,
                "is_crisis": response.is_crisis
            })

        finally:
            db.close()

    except Exception as e:
        logger.error(f"✗ awe-chat error: {e}")
        return JSONResponse({
            "response": "Xin lỗi, có lỗi xảy ra.",
            "error": True
        }, status_code=500)


# --------------------------------------------------------------
# RUN SERVER
# --------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "server_enhanced:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT", "production") == "development"
    )
