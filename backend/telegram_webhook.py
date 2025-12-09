"""
AWE Mental Health Chatbot - Telegram Webhook Integration

For production deployment (recommended for cloud platforms like Azure, AWS, Heroku)
Uses FastAPI + python-telegram-bot with webhook pattern

Features:
- Webhook-based message handling (more efficient than polling)
- FastAPI integration with existing web server
- Scalable for production environments
- Works seamlessly with FastAPI's async architecture
"""

import os
import logging
import hmac
import hashlib
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from telegram import Update, Bot
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Database setup
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from database_aad import get_database_engine
from database import Base, set_engine_and_session

# Chatbot + RAG
from chatbot import TherapeuticChatbot
from rag_system_v2 import TherapeuticRAG

# --------------------------------------------------------------
# LOGGING CONFIGURATION
# --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress verbose Telegram logs
logging.getLogger("telegram.vendor.requests.packages.urllib3.connectionpool").setLevel(
    logging.WARNING
)

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
# TELEGRAM CONFIGURATION
# --------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL")  # e.g., https://yourdomain.com/api/telegram

if not TELEGRAM_BOT_TOKEN:
    raise ValueError(
        "❌ TELEGRAM_BOT_TOKEN not set! "
        "Please set it in your .env file or environment variables."
    )

logger.info(f"✓ Telegram bot token loaded")

# Global references
chatbot: Optional[TherapeuticChatbot] = None
rag_system: Optional[TherapeuticRAG] = None
telegram_app: Optional[Application] = None
telegram_bot: Optional[Bot] = None


# --------------------------------------------------------------
# INITIALIZATION FUNCTION
# --------------------------------------------------------------
async def initialize_telegram_bot():
    """Initialize Telegram bot application."""
    global chatbot, rag_system, telegram_app, telegram_bot

    logger.info("🚀 Initializing Telegram bot...")

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
    try:
        logger.info("📚 Initializing knowledge base (RAG system)...")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            logger.warning("⚠️ OPENAI_API_KEY not set - RAG system will not be initialized")
        else:
            rag_system = TherapeuticRAG(openai_api_key=openai_api_key)
            logger.info("✓ Knowledge base initialized")

    except Exception as e:
        logger.warning(f"⚠️ RAG system initialization failed: {e}")
        rag_system = None

    # Chatbot Initialization
    try:
        logger.info("🤖 Initializing therapeutic chatbot...")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            logger.error("✗ OPENAI_API_KEY not set!")
            raise ValueError("OPENAI_API_KEY is required")

        chatbot = TherapeuticChatbot(
            openai_api_key=openai_api_key,
            rag_system=rag_system
        )

        logger.info("✓ Chatbot initialized successfully")

    except Exception as e:
        logger.error(f"✗ Chatbot initialization failed: {e}")
        raise

    # Initialize Telegram Application
    try:
        logger.info("📱 Setting up Telegram bot application...")

        telegram_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

        # Add command handlers
        telegram_app.add_handler(CommandHandler("start", start))
        telegram_app.add_handler(CommandHandler("help", help_command))
        telegram_app.add_handler(CommandHandler("crisis", crisis_command))

        # Add message handler
        telegram_app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
        )

        # Add error handler
        telegram_app.add_error_handler(error_handler)

        # Get bot instance
        telegram_bot = telegram_app.bot

        logger.info(f"✓ Telegram bot app initialized (Bot: @{telegram_bot.username})")

    except Exception as e:
        logger.error(f"✗ Failed to initialize Telegram app: {e}")
        raise

    logger.info("""
╔══════════════════════════════════════════════════════════════╗
║  AWE Therapeutic Chatbot - Telegram Webhook Integration       ║
║              Production-Ready Deployment                       ║
╚══════════════════════════════════════════════════════════════╝
""")

    logger.info("🎉 Telegram bot initialization complete!")
    logger.info(f"📲 Webhook URL: {TELEGRAM_WEBHOOK_URL if TELEGRAM_WEBHOOK_URL else 'Not configured'}")


# --------------------------------------------------------------
# COMMAND HANDLERS
# --------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    user = update.effective_user
    logger.info(f"👤 New user started bot: {user.first_name} (ID: {user.id})")

    welcome_message = (
        f"Hello {user.first_name}! 👋\n\n"
        "I'm your mental health assistant from AWE (Awareness-App). "
        "I'm here to provide supportive conversations about digital wellness, "
        "mental health, and emotional well-being.\n\n"
        "🌟 **What I can help with:**\n"
        "• Screen time and digital health concerns\n"
        "• Stress and anxiety management\n"
        "• Habit tracking and mindfulness\n"
        "• General wellness support\n\n"
        "📱 Just send me a message to start our conversation.\n"
        "If you're in crisis, I'll provide resources to professional support.\n\n"
        "Let's start — how are you feeling today?"
    )

    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    help_message = (
        "🆘 **Help & Support**\n\n"
        "I'm here to provide supportive conversations.\n\n"
        "**Commands:**\n"
        "/start - Start conversation\n"
        "/help - Show this message\n"
        "/crisis - Get crisis resources\n\n"
        "**Just send me a message** - I'll respond with therapeutic support.\n\n"
        "If you're experiencing a mental health crisis, please reach out to professionals:\n"
        "• National Suicide Prevention Lifeline: 988\n"
        "• Crisis Text Line: Text HOME to 741741\n"
        "• SAMHSA National Helpline: 1-800-662-4357"
    )

    await update.message.reply_text(help_message)


async def crisis_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /crisis command."""
    crisis_message = (
        "🆘 **If You're In Crisis**\n\n"
        "I hear that you may need immediate support. Your safety is important.\n\n"
        "**Immediate Resources:**\n"
        "🚨 National Suicide Prevention Lifeline: **988**\n"
        "(Call or text, available 24/7)\n\n"
        "💬 Crisis Text Line: Text **HOME** to **741741**\n"
        "(Free, confidential, available 24/7)\n\n"
        "📞 SAMHSA National Helpline: **1-800-662-4357**\n"
        "(Free, confidential, 24 hours a day)\n\n"
        "🏥 **If in immediate danger:** Call 911 or go to the nearest emergency room.\n\n"
        "Remember: You're not alone, and help is available right now."
    )

    await update.message.reply_text(crisis_message)


# --------------------------------------------------------------
# MESSAGE HANDLER
# --------------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle incoming text messages."""
    user = update.effective_user
    user_message = update.message.text.strip()

    if not user_message:
        return

    logger.info(
        f"💬 Message from {user.first_name} (ID: {user.id}): {user_message[:50]}..."
    )

    try:
        db = SessionLocal()

        try:
            if not chatbot:
                await update.message.reply_text(
                    "❌ I apologize, but the chatbot is not properly initialized. "
                    "Please try again in a moment."
                )
                return

            # Generate response
            telegram_user_id = f"telegram:{user.id}"

            response_dict = chatbot.generate_response(
                db=db,
                whatsapp_number=telegram_user_id,
                user_message=user_message
            )

            response_text = response_dict.get("response", "")
            is_crisis = response_dict.get("is_crisis", False)

            logger.info(f"✓ Response generated (Crisis: {is_crisis})")

            # Send response
            await update.message.reply_text(response_text)

            if is_crisis:
                logger.warning(f"⚠️ Crisis content detected from user {user.id}")

        except Exception as e:
            logger.error(f"✗ Error processing message: {e}", exc_info=True)

            fallback_response = (
                "I apologize, but I'm having technical issues right now.\n\n"
                "If you're in crisis, please contact:\n"
                "• 988 Suicide & Crisis Lifeline\n"
                "• Text HOME to 741741\n\n"
                "Please try again in a moment."
            )

            await update.message.reply_text(fallback_response)

        finally:
            db.close()

    except Exception as e:
        logger.error(f"✗ Critical error: {e}", exc_info=True)


# --------------------------------------------------------------
# ERROR HANDLER
# --------------------------------------------------------------
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors caused by Updates."""
    logger.error(f"Update {update} caused error: {context.error}", exc_info=True)


# --------------------------------------------------------------
# FASTAPI ROUTES
# --------------------------------------------------------------
async def create_telegram_router(app: FastAPI):
    """
    Create and register Telegram webhook routes to FastAPI app.
    Call this during app startup.
    """

    await initialize_telegram_bot()

    # Verify webhook signature (security)
    def verify_telegram_signature(data: bytes, signature: str) -> bool:
        """Verify that the request came from Telegram."""
        computed_hash = hmac.new(
            TELEGRAM_BOT_TOKEN.encode(),
            data,
            hashlib.sha256
        ).hexdigest()
        return computed_hash == signature

    @app.post("/api/telegram")
    async def telegram_webhook(request: Request):
        """Telegram webhook endpoint."""
        try:
            # Verify signature
            signature = request.headers.get("X-Telegram-Bot-Api-Secret-Hash", "")
            body = await request.body()

            if not verify_telegram_signature(body, signature):
                logger.warning("⚠️ Invalid Telegram signature")
                return JSONResponse({"status": "invalid_signature"}, status_code=403)

            # Process update
            json_data = await request.json()
            update = Update.de_json(json_data, telegram_bot)

            if update:
                logger.info(f"📨 Received Telegram update: {update.update_id}")
                await telegram_app.process_update(update)

            return JSONResponse({"status": "ok"})

        except Exception as e:
            logger.error(f"✗ Telegram webhook error: {e}", exc_info=True)
            return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)

    @app.get("/api/telegram/info")
    async def telegram_info():
        """Get Telegram bot info."""
        try:
            if telegram_bot:
                return {
                    "status": "active",
                    "bot_username": telegram_bot.username,
                    "webhook_url": TELEGRAM_WEBHOOK_URL,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "status": "inactive",
                    "message": "Telegram bot not initialized"
                }
        except Exception as e:
            logger.error(f"Error getting telegram info: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/telegram/set-webhook")
    async def set_webhook():
        """Manually set Telegram webhook (call once during deployment)."""
        try:
            if not TELEGRAM_WEBHOOK_URL:
                return JSONResponse({
                    "status": "error",
                    "message": "TELEGRAM_WEBHOOK_URL not configured"
                }, status_code=400)

            if not telegram_bot:
                return JSONResponse({
                    "status": "error",
                    "message": "Telegram bot not initialized"
                }, status_code=500)

            webhook_info = await telegram_bot.set_webhook(
                url=TELEGRAM_WEBHOOK_URL,
                allowed_updates=["message", "edited_channel_post", "callback_query"],
                secret_token=TELEGRAM_BOT_TOKEN
            )

            logger.info(f"✓ Webhook set: {TELEGRAM_WEBHOOK_URL}")

            return JSONResponse({
                "status": "success",
                "message": "Webhook configured",
                "url": TELEGRAM_WEBHOOK_URL
            })

        except Exception as e:
            logger.error(f"Error setting webhook: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/telegram/webhook-info")
    async def get_webhook_info():
        """Get current webhook information."""
        try:
            if not telegram_bot:
                return JSONResponse({"status": "error", "message": "Bot not initialized"})

            webhook_info = await telegram_bot.get_webhook_info()

            return JSONResponse({
                "url": webhook_info.url,
                "has_custom_certificate": webhook_info.has_custom_certificate,
                "pending_update_count": webhook_info.pending_update_count,
                "last_error_date": webhook_info.last_error_date,
                "last_error_message": webhook_info.last_error_message
            })

        except Exception as e:
            logger.error(f"Error getting webhook info: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    logger.info("✓ Telegram webhook routes registered")
