"""
Telegram Webhook Server Enhanced - Hỗ Trợ Sức Khỏe Tâm Thần

Phiên bản Webhook với Enhanced Chatbot (8 Layers)
Sử dụng cho production deployment với Azure/Heroku/etc.

Tính năng:
- Webhook endpoint thay vì long polling
- Tích hợp đầy đủ 8 layers nâng cao
- Rate limiting và error handling
- Health check endpoint
- Proactive follow-up notifications
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import nest_asyncio
from fastapi import FastAPI, Request, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from telegram import Update, Bot, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Database
from sqlalchemy.orm import sessionmaker, Session
from database_aad import get_database_engine
from database import Base, set_engine_and_session

# Enhanced Chatbot
from enhanced_chatbot import EnhancedTherapeuticChatbot, create_enhanced_chatbot
from rag_system_v2 import TherapeuticRAG
from services.rate_limiter import RateLimiter

# Áp dụng nest_asyncio
nest_asyncio.apply()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Cấu hình từ biến môi trường
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
WEBHOOK_URL = os.getenv('WEBHOOK_URL', '')  # e.g., https://your-app.azurewebsites.net/webhook
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET', '')

# Kiểm tra token
if not TELEGRAM_BOT_TOKEN:
    logger.error("❌ TELEGRAM_BOT_TOKEN chưa được đặt")
    raise ValueError("Cần TELEGRAM_BOT_TOKEN")

# Thiết lập cơ sở dữ liệu
engine = get_database_engine()
session_factory = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)

set_engine_and_session(engine, session_factory)

# Biến toàn cục
chatbot: Optional[EnhancedTherapeuticChatbot] = None
rag_system: Optional[TherapeuticRAG] = None
telegram_app: Optional[Application] = None
rate_limiter = RateLimiter(max_requests=30, window_seconds=60)


# ============================================================
# DATABASE DEPENDENCY
# ============================================================

def get_db():
    """Dependency để lấy database session."""
    db = session_factory()
    try:
        yield db
    finally:
        db.close()


# ============================================================
# LIFESPAN HANDLERS
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo và dọn dẹp khi app start/stop."""
    global chatbot, rag_system, telegram_app
    
    logger.info("=" * 70)
    logger.info("║    Telegram Webhook Server - ENHANCED VERSION           ║")
    logger.info("║    Hỗ Trợ Sức Khỏe Tâm Thần với 8 Layers Nâng Cao       ║")
    logger.info("=" * 70)
    
    # Khởi tạo RAG
    try:
        logger.info("📚 Khởi tạo hệ thống kiến thức (RAG)...")
        rag_system = TherapeuticRAG(google_api_key=GOOGLE_API_KEY)
        logger.info("✓ RAG khởi tạo thành công")
    except Exception as e:
        logger.warning(f"⚠️ Không khởi tạo RAG: {e}")
    
    # Khởi tạo Enhanced Chatbot
    try:
        logger.info("🤖 Khởi tạo Enhanced Chatbot với 8 layers...")
        chatbot = create_enhanced_chatbot(
            google_api_key=GOOGLE_API_KEY,
            rag_system=rag_system,
            db_session_factory=session_factory
        )
        
        logger.info("✓ Enhanced Chatbot khởi tạo thành công")
        logger.info("  ├─ Layer 1-4: Conversational, Personalization, Emotion, Storytelling")
        logger.info("  └─ Layer 5-8: RAG Precision, Reasoning, Safety, Proactive")
        
    except Exception as e:
        logger.error(f"✗ Khởi tạo Enhanced Chatbot thất bại: {e}")
        raise

    # Khởi tạo Telegram Application
    try:
        telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Thêm handlers
        telegram_app.add_handler(CommandHandler("start", start_handler))
        telegram_app.add_handler(CommandHandler("help", help_handler))
        telegram_app.add_handler(CommandHandler("crisis", crisis_handler))
        telegram_app.add_handler(CommandHandler("story", story_handler))
        telegram_app.add_handler(CommandHandler("exercise", exercise_handler))
        telegram_app.add_handler(CommandHandler("summary", summary_handler))
        telegram_app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
        )
        
        # Khởi tạo bot
        await telegram_app.initialize()
        
        # Set commands
        commands = [
            BotCommand("start", "Bắt đầu cuộc trò chuyện"),
            BotCommand("help", "Xem trợ giúp"),
            BotCommand("crisis", "Tài nguyên khủng hoảng"),
            BotCommand("story", "Nhận câu chuyện trị liệu"),
            BotCommand("exercise", "Nhận bài tập thư giãn"),
            BotCommand("summary", "Xem tóm tắt"),
        ]
        await telegram_app.bot.set_my_commands(commands)
        
        # Đăng ký webhook nếu có URL
        if WEBHOOK_URL:
            webhook_path = f"{WEBHOOK_URL}/webhook"
            await telegram_app.bot.set_webhook(
                url=webhook_path,
                secret_token=WEBHOOK_SECRET if WEBHOOK_SECRET else None
            )
            logger.info(f"✓ Webhook đăng ký: {webhook_path}")
        
        logger.info("✓ Telegram Application sẵn sàng")
        
    except Exception as e:
        logger.error(f"✗ Khởi tạo Telegram thất bại: {e}")
        raise
    
    logger.info("=" * 70)
    logger.info("✅ Server sẵn sàng nhận webhook!")
    logger.info("=" * 70 + "\n")
    
    yield
    
    # Shutdown
    logger.info("⏹️ Đang dừng server...")
    try:
        if telegram_app:
            await telegram_app.bot.delete_webhook()
            await telegram_app.shutdown()
        engine.dispose()
        logger.info("✓ Cleanup hoàn thành")
    except Exception as e:
        logger.error(f"⚠️ Shutdown error: {e}")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="AWE Telegram Webhook - Enhanced",
    description="Webhook server với Enhanced Chatbot (8 Layers)",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================
# TELEGRAM HANDLERS
# ============================================================

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler cho /start."""
    user = update.effective_user
    user_id = f"telegram:{user.id}"
    
    if chatbot:
        greeting = chatbot.generate_greeting(user_id)
    else:
        greeting = f"Xin chào {user.first_name}!"
    
    welcome = (
        f"{greeting}\n\n"
        "Tôi là trợ lý sức khỏe tâm thần từ AWE với AI nâng cao.\n"
        "✨ Nhận diện cảm xúc | 💝 Cá nhân hóa | 📖 Kể chuyện trị liệu\n\n"
        "Bạn đang cảm thấy thế nào hôm nay? 💚"
    )
    
    await update.message.reply_text(welcome)


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler cho /help."""
    help_text = (
        "🆘 **Trợ Giúp**\n\n"
        "/start - Bắt đầu\n"
        "/help - Trợ giúp\n"
        "/crisis - Tài nguyên khủng hoảng\n"
        "/story [vấn đề] - Câu chuyện trị liệu\n"
        "/exercise [loại] - Bài tập thư giãn\n"
        "/summary - Tóm tắt cuộc trò chuyện\n\n"
        "📞 Đường dây nóng: 1800 599 920"
    )
    await update.message.reply_text(help_text)


async def crisis_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler cho /crisis."""
    crisis_text = (
        "🆘 **Tài Nguyên Khủng Hoảng**\n\n"
        "📞 Đường dây nóng: 1800 599 920 (24/7)\n"
        "🏥 Cấp cứu: 115\n"
        "👮 Công an: 113\n\n"
        "Bạn không đơn độc. Hãy gọi ngay nếu cần hỗ trợ. 💚"
    )
    await update.message.reply_text(crisis_text)


async def story_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler cho /story."""
    user = update.effective_user
    user_id = f"telegram:{user.id}"
    
    await context.bot.send_chat_action(update.effective_chat.id, "typing")
    
    if not chatbot:
        await update.message.reply_text("❌ Chatbot chưa sẵn sàng.")
        return
    
    issue = " ".join(context.args) if context.args else "căng thẳng"
    
    try:
        story = chatbot.storytelling.generate_story(
            issue=issue,
            context="",
            emotion="",
            approach=None
        )
        await update.message.reply_text(story)
    except Exception as e:
        logger.error(f"Story error: {e}")
        await update.message.reply_text("Xin lỗi, có lỗi khi tạo câu chuyện.")


async def exercise_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler cho /exercise."""
    user = update.effective_user
    user_id = f"telegram:{user.id}"
    
    if not chatbot:
        await update.message.reply_text("❌ Chatbot chưa sẵn sàng.")
        return
    
    issue = " ".join(context.args) if context.args else None
    
    try:
        exercise = chatbot.get_therapeutic_exercise(user_id, issue)
        await update.message.reply_text(exercise)
    except Exception as e:
        logger.error(f"Exercise error: {e}")
        await update.message.reply_text("Xin lỗi, có lỗi khi lấy bài tập.")


async def summary_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler cho /summary."""
    user = update.effective_user
    user_id = f"telegram:{user.id}"
    
    if not chatbot:
        await update.message.reply_text("❌ Chatbot chưa sẵn sàng.")
        return
    
    try:
        summary = chatbot.get_conversation_summary(user_id)
        proactive = summary.get("proactive", {})
        
        summary_text = (
            f"📊 **Tóm Tắt**\n\n"
            f"• Trạng thái: {proactive.get('current_state', 'N/A')}\n"
            f"• Tin nhắn: {proactive.get('messages_count', 0)}\n"
            f"• Cảm xúc: {proactive.get('last_emotion', 'Chưa xác định')}\n"
            f"• Chủ đề: {', '.join(proactive.get('session_topics', [])) or 'Chưa có'}"
        )
        await update.message.reply_text(summary_text)
    except Exception as e:
        logger.error(f"Summary error: {e}")
        await update.message.reply_text("Xin lỗi, không thể lấy tóm tắt.")


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler cho tin nhắn văn bản."""
    user = update.effective_user
    user_message = update.message.text.strip()
    
    if not user_message:
        return
    
    user_id = f"telegram:{user.id}"
    
    logger.info(f"💬 [{user.first_name}] {user_message[:50]}...")
    
    # Rate limiting
    if not rate_limiter.allow_request(user_id):
        await update.message.reply_text(
            "⏳ Bạn đang gửi tin nhắn quá nhanh. Vui lòng đợi một chút."
        )
        return
    
    await context.bot.send_chat_action(update.effective_chat.id, "typing")
    
    if not chatbot:
        await update.message.reply_text(
            "❌ Chatbot chưa sẵn sàng. Vui lòng thử lại sau."
        )
        return
    
    try:
        db = session_factory()
        try:
            response = chatbot.generate_response(
                db=db,
                user_id=user_id,
                user_message=user_message,
                use_rag=True,
                use_storytelling=False,
                use_proactive=True
            )
            
            logger.info(
                f"✓ Response - Emotion: {response.emotion_detected}, "
                f"Crisis: {response.is_crisis}"
            )
            
            await update.message.reply_text(response.response)
            
            if response.is_crisis:
                logger.warning(f"⚠️ CRISIS detected: {user_id}")
                
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Message error: {e}", exc_info=True)
        await update.message.reply_text(
            "Xin lỗi, có lỗi xảy ra.\n"
            "📞 Đường dây nóng: 1800 599 920"
        )


# ============================================================
# WEBHOOK ENDPOINTS
# ============================================================

@app.post("/webhook")
async def telegram_webhook(request: Request):
    """Nhận webhook từ Telegram."""
    if not telegram_app:
        raise HTTPException(status_code=503, detail="Bot chưa sẵn sàng")
    
    # Xác thực secret token nếu có
    if WEBHOOK_SECRET:
        token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if token != WEBHOOK_SECRET:
            raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        
        # Xử lý update
        await telegram_app.process_update(update)
        
        return JSONResponse({"status": "ok"})
        
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "chatbot_ready": chatbot is not None,
        "rag_ready": rag_system is not None,
        "telegram_ready": telegram_app is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "AWE Mental Health Chatbot - Enhanced",
        "version": "2.0.0",
        "features": [
            "8-Layer Enhanced AI",
            "Emotion Recognition",
            "Personalization",
            "Storytelling Therapy",
            "RAG Precision Boost",
            "Crisis Detection",
            "Proactive Dialogue"
        ],
        "endpoints": {
            "webhook": "/webhook",
            "health": "/health"
        }
    }


# ============================================================
# PROACTIVE NOTIFICATIONS (Background Task)
# ============================================================

class ProactiveNotification(BaseModel):
    user_id: str
    message: str


@app.post("/api/send-proactive")
async def send_proactive_message(
    notification: ProactiveNotification,
    background_tasks: BackgroundTasks
):
    """Gửi tin nhắn proactive đến user."""
    if not telegram_app:
        raise HTTPException(status_code=503, detail="Bot chưa sẵn sàng")
    
    async def send_message():
        try:
            # Extract telegram ID from user_id
            if notification.user_id.startswith("telegram:"):
                telegram_id = int(notification.user_id.replace("telegram:", ""))
            else:
                telegram_id = int(notification.user_id)
            
            await telegram_app.bot.send_message(
                chat_id=telegram_id,
                text=notification.message
            )
            logger.info(f"✓ Proactive message sent to {notification.user_id}")
            
        except Exception as e:
            logger.error(f"Failed to send proactive: {e}")
    
    background_tasks.add_task(send_message)
    
    return {"status": "scheduled", "user_id": notification.user_id}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting Enhanced Webhook Server on {host}:{port}")
    
    uvicorn.run(
        "telegram_webhook_enhanced:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
