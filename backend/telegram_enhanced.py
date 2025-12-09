"""
Bot Telegram AWE - Hỗ Trợ Sức Khỏe Tâm Thần (ENHANCED VERSION)

Tích hợp đầy đủ 8 layers nâng cao:
1. Conversational Layer - Biến RAG thành hội thoại tự nhiên
2. Personalization Layer - Cá nhân hóa theo người dùng
3. Emotional Understanding Layer - Nhận diện cảm xúc
4. Storytelling Therapy Mode - Tạo câu chuyện trị liệu
5. RAG Precision Boost - Multi-query, Hybrid, Reranking
6. Reasoning Layer - Chain-of-Thought, Self-Refinement
7. Safety & Ethics Layer - Kiểm tra an toàn
8. Proactive Dialogue Engine - Chủ động hỏi thăm

Tính năng:
- Xử lý lệnh /start, /help, /crisis, /story, /exercise
- Xử lý tin nhắn với ngữ cảnh RAG nâng cao
- Phát hiện và xử lý khủng hoảng
- Nhận diện cảm xúc và cá nhân hóa
- Theo dõi cuộc hội thoại và proactive follow-up
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Optional

import nest_asyncio
from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Thiết lập cơ sở dữ liệu
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from database_aad import get_database_engine
from database import Base, set_engine_and_session

# Enhanced Chatbot + RAG
from enhanced_chatbot import EnhancedTherapeuticChatbot, create_enhanced_chatbot
from rag_system_v2 import TherapeuticRAG

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

# Thiết lập cơ sở dữ liệu
engine = get_database_engine()
session_factory = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)

set_engine_and_session(engine, session_factory)
SessionLocal = session_factory

# Biến toàn cục
chatbot: Optional[EnhancedTherapeuticChatbot] = None
rag_system: Optional[TherapeuticRAG] = None

# Kiểm tra token
if not TELEGRAM_BOT_TOKEN:
    logger.error("❌ TELEGRAM_BOT_TOKEN chưa được đặt")
    raise ValueError("Cần TELEGRAM_BOT_TOKEN trong .env")


async def startup():
    """Khởi tạo Enhanced Chatbot khi bot bắt đầu."""
    global chatbot, rag_system
    
    logger.info("=" * 70)
    logger.info("║    Bot Telegram AWE - ENHANCED VERSION                  ║")
    logger.info("║    Hỗ Trợ Sức Khỏe Tâm Thần với 8 Layers Nâng Cao       ║")
    logger.info("=" * 70)
    
    # Khởi tạo RAG
    try:
        logger.info("📚 Khởi tạo hệ thống kiến thức (RAG)...")
        rag_system = TherapeuticRAG(google_api_key=GOOGLE_API_KEY)
        logger.info("✓ RAG khởi tạo thành công")
    except Exception as e:
        logger.warning(f"⚠️ Không khởi tạo RAG: {e}")
    
    # Khởi tạo Enhanced Chatbot với tất cả layers
    try:
        logger.info("🤖 Khởi tạo Enhanced Chatbot với 8 layers...")
        chatbot = create_enhanced_chatbot(
            google_api_key=GOOGLE_API_KEY,
            rag_system=rag_system,
            db_session_factory=session_factory
        )
        
        logger.info("✓ Enhanced Chatbot khởi tạo thành công:")
        logger.info("  ├─ Layer 1: Conversational Layer ✓")
        logger.info("  ├─ Layer 2: Personalization Layer ✓")
        logger.info("  ├─ Layer 3: Emotional Understanding ✓")
        logger.info("  ├─ Layer 4: Storytelling Therapy ✓")
        logger.info("  ├─ Layer 5: RAG Precision Boost ✓")
        logger.info("  ├─ Layer 6: Reasoning Layer ✓")
        logger.info("  ├─ Layer 7: Safety & Ethics ✓")
        logger.info("  └─ Layer 8: Proactive Dialogue ✓")
        
    except Exception as e:
        logger.error(f"✗ Khởi tạo Enhanced Chatbot thất bại: {e}")
        raise

    logger.info("=" * 70)
    logger.info("✅ Bot sẵn sàng với tất cả tính năng nâng cao!")
    logger.info("=" * 70 + "\n")


async def shutdown():
    """Dọn dẹp khi bot dừng."""
    logger.info("⏹️ Bot dừng lại")
    try:
        engine.dispose()
        logger.info("✓ Database connections closed")
    except Exception as e:
        logger.error(f"⚠️ Shutdown error: {e}")


# ============================================================
# COMMAND HANDLERS
# ============================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý lệnh /start."""
    user = update.effective_user
    user_id = f"telegram:{user.id}"
    
    # Lấy lời chào cá nhân hóa từ Enhanced Chatbot
    if chatbot:
        greeting = chatbot.generate_greeting(user_id)
    else:
        greeting = f"Xin chào {user.first_name}! 👋"
    
    welcome_message = (
        f"{greeting}\n\n"
        "Tôi là trợ lý sức khỏe tâm thần từ AWE - phiên bản nâng cao với AI thông minh. "
        "Tôi có thể hiểu cảm xúc, ghi nhớ cuộc trò chuyện, và hỗ trợ bạn theo cách cá nhân hóa.\n\n"
        "✨ **Tính năng đặc biệt:**\n"
        "• 🎭 Nhận diện cảm xúc từ tin nhắn\n"
        "• 💝 Cá nhân hóa theo bạn\n"
        "• 📖 Kể chuyện trị liệu (/story)\n"
        "• 🧘 Bài tập thư giãn (/exercise)\n"
        "• 🔍 Tìm kiếm thông tin chính xác\n"
        "• 🛡️ An toàn và đạo đức\n\n"
        "📱 Hãy chia sẻ với tôi - bạn đang cảm thấy thế nào?\n\n"
        "Lệnh: /help, /crisis, /story, /exercise"
    )
    
    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý lệnh /help."""
    help_message = (
        "🆘 **Trợ Giúp & Hỗ Trợ**\n\n"
        "Tôi là chatbot hỗ trợ sức khỏe tâm thần với AI nâng cao.\n\n"
        "**Lệnh có sẵn:**\n"
        "/start - Bắt đầu cuộc trò chuyện\n"
        "/help - Hiển thị trợ giúp này\n"
        "/crisis - Nhận tài nguyên khủng hoảng\n"
        "/story - Nhận câu chuyện trị liệu\n"
        "/exercise - Nhận bài tập thư giãn\n"
        "/summary - Xem tóm tắt cuộc trò chuyện\n\n"
        "**Cách sử dụng:**\n"
        "• Chỉ cần gửi tin nhắn - tôi sẽ phản hồi với sự đồng cảm\n"
        "• Chia sẻ cảm xúc - tôi sẽ nhận diện và điều chỉnh giọng điệu\n"
        "• Tôi ghi nhớ cuộc trò chuyện và có thể hỏi thăm bạn\n\n"
        "**Liên hệ khẩn cấp:**\n"
        "• Đường dây nóng: 1800 599 920\n"
        "• Cấp cứu: 115\n\n"
        "Sức khỏe của bạn rất quan trọng! 💚"
    )
    
    await update.message.reply_text(help_message)


async def crisis_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý lệnh /crisis."""
    crisis_message = (
        "🆘 **Nếu Bạn Đang Gặp Khủng Hoảng**\n\n"
        "Tôi hiểu bạn có thể cần hỗ trợ ngay lập tức. An toàn của bạn là quan trọng nhất.\n\n"
        "**📞 Liên Hệ Khẩn Cấp:**\n\n"
        "🚨 **Đường dây nóng hỗ trợ tâm lý:**\n"
        "• 1800 599 920 (miễn phí, 24/7)\n"
        "• 1900 0027 (Tổng đài sức khỏe tâm thần)\n\n"
        "💬 **Cấp cứu y tế:** 115\n"
        "👮 **Công an:** 113\n\n"
        "🏥 **Nếu gặp nguy hiểm tức thì:**\n"
        "Đến bệnh viện gần nhất hoặc gọi 115\n\n"
        "💚 Nhớ rằng: Bạn không đơn độc. "
        "Những chuyên gia được đào tạo sẵn sàng lắng nghe và hỗ trợ bạn.\n\n"
        "Tôi ở đây bên bạn. 🌿"
    )
    
    await update.message.reply_text(crisis_message)


async def story_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý lệnh /story - Tạo câu chuyện trị liệu."""
    user = update.effective_user
    user_id = f"telegram:{user.id}"
    
    # Hiển thị trạng thái đang gõ
    await context.bot.send_chat_action(update.effective_chat.id, "typing")
    
    if not chatbot:
        await update.message.reply_text("❌ Chatbot chưa sẵn sàng. Vui lòng thử lại.")
        return
    
    # Lấy vấn đề từ argument hoặc từ memory
    args = context.args
    if args:
        issue = " ".join(args)
    else:
        issue = None
    
    try:
        # Sử dụng storytelling module
        story = chatbot.storytelling.generate_story(
            issue=issue or "căng thẳng",
            context="",
            emotion="",
            approach=None
        )
        
        await update.message.reply_text(story)
        
    except Exception as e:
        logger.error(f"❌ Lỗi tạo story: {e}")
        await update.message.reply_text(
            "Xin lỗi, có lỗi khi tạo câu chuyện. Vui lòng thử lại."
        )


async def exercise_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý lệnh /exercise - Lấy bài tập trị liệu."""
    user = update.effective_user
    user_id = f"telegram:{user.id}"
    
    if not chatbot:
        await update.message.reply_text("❌ Chatbot chưa sẵn sàng. Vui lòng thử lại.")
        return
    
    # Lấy vấn đề từ argument
    args = context.args
    issue = " ".join(args) if args else None
    
    try:
        exercise = chatbot.get_therapeutic_exercise(user_id, issue)
        await update.message.reply_text(exercise)
        
    except Exception as e:
        logger.error(f"❌ Lỗi lấy exercise: {e}")
        await update.message.reply_text(
            "Xin lỗi, có lỗi khi lấy bài tập. Vui lòng thử lại."
        )


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý lệnh /summary - Xem tóm tắt cuộc hội thoại."""
    user = update.effective_user
    user_id = f"telegram:{user.id}"
    
    if not chatbot:
        await update.message.reply_text("❌ Chatbot chưa sẵn sàng.")
        return
    
    try:
        summary = chatbot.get_conversation_summary(user_id)
        
        proactive = summary.get("proactive", {})
        personal = summary.get("personalization", {})
        
        summary_text = (
            "📊 **Tóm Tắt Cuộc Trò Chuyện**\n\n"
            f"• Trạng thái: {proactive.get('current_state', 'N/A')}\n"
            f"• Số tin nhắn: {proactive.get('messages_count', 0)}\n"
            f"• Cảm xúc gần nhất: {proactive.get('last_emotion', 'Chưa xác định')}\n"
            f"• Chủ đề đã thảo luận: {', '.join(proactive.get('session_topics', [])) or 'Chưa có'}\n"
            f"• Cần theo dõi: {', '.join(proactive.get('pending_follow_ups', [])) or 'Không có'}\n\n"
            f"💚 Mình ở đây lắng nghe bạn!"
        )
        
        await update.message.reply_text(summary_text)
        
    except Exception as e:
        logger.error(f"❌ Lỗi lấy summary: {e}")
        await update.message.reply_text(
            "Xin lỗi, không thể lấy tóm tắt. Vui lòng thử lại."
        )


# ============================================================
# MESSAGE HANDLER - CORE ENHANCED PROCESSING
# ============================================================

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý tin nhắn văn bản với Enhanced Chatbot."""
    user = update.effective_user
    chat_id = update.effective_chat.id
    user_message = update.message.text.strip()
    
    if not user_message:
        return
    
    user_id = f"telegram:{user.id}"
    
    try:
        logger.info(f"💬 Tin nhắn từ {user.first_name} (ID: {user.id}): {user_message[:50]}...")
        
        # Hiển thị trạng thái đang gõ
        try:
            await context.bot.send_chat_action(chat_id, "typing")
        except Exception as e:
            logger.warning(f"⚠️ Lỗi gửi trạng thái gõ: {e}")
        
        # Kiểm tra chatbot
        if not chatbot:
            await update.message.reply_text(
                "❌ Xin lỗi, chatbot chưa được khởi tạo đúng cách. "
                "Vui lòng thử lại sau vài giây."
            )
            logger.error("Enhanced Chatbot chưa khởi tạo")
            return
        
        # Tạo phản hồi bằng Enhanced Chatbot
        try:
            db = SessionLocal()
            try:
                # Sử dụng Enhanced generate_response với tất cả layers
                response = chatbot.generate_response(
                    db=db,
                    user_id=user_id,
                    user_message=user_message,
                    use_rag=True,
                    use_storytelling=False,  # Chỉ dùng khi /story
                    use_proactive=True
                )
                
                response_text = response.response
                is_crisis = response.is_crisis
                emotion = response.emotion_detected
                intensity = response.emotion_intensity
                
                logger.info(
                    f"✓ Enhanced response - Emotion: {emotion} ({intensity}), "
                    f"Crisis: {is_crisis}, RAG: {response.used_rag}"
                )
                
                # Gửi phản hồi
                await update.message.reply_text(response_text)
                
                # Log chi tiết nếu phát hiện khủng hoảng
                if is_crisis:
                    logger.warning(
                        f"⚠️ KHỦNG HOẢNG phát hiện từ {user.first_name} (ID: {user.id})"
                    )
                
                # Log proactive elements nếu có
                if response.proactive_elements:
                    logger.info(f"💡 Proactive elements: {response.proactive_elements}")
                    
            finally:
                db.close()
        
        except Exception as e:
            logger.error(f"✗ Lỗi xử lý tin nhắn: {e}", exc_info=True)
            
            # Phản hồi dự phòng
            fallback_response = (
                "Xin lỗi, mình gặp chút trục trặc kỹ thuật. 🙏\n\n"
                "Nếu bạn cần hỗ trợ khẩn cấp:\n"
                "• Đường dây nóng: 1800 599 920\n"
                "• Cấp cứu: 115\n\n"
                "Vui lòng thử lại hoặc gõ /help để xem trợ giúp."
            )
            await update.message.reply_text(fallback_response)
    
    except Exception as e:
        logger.error(f"❌ Lỗi nghiêm trọng: {e}", exc_info=True)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý lỗi."""
    logger.error(f"Update gây ra lỗi: {context.error}", exc_info=True)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main() -> None:
    """Bắt đầu bot Telegram với Enhanced Chatbot."""
    
    # Tạo ứng dụng
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Thêm trình xử lý lệnh
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("crisis", crisis_command))
    application.add_handler(CommandHandler("story", story_command))
    application.add_handler(CommandHandler("exercise", exercise_command))
    application.add_handler(CommandHandler("summary", summary_command))
    
    # Thêm trình xử lý tin nhắn
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    
    # Thêm trình xử lý lỗi
    application.add_error_handler(error_handler)
    
    # Khởi tạo startup/shutdown
    async def bot_startup(app):
        await startup()
        
        # Set bot commands
        commands = [
            BotCommand("start", "Bắt đầu cuộc trò chuyện"),
            BotCommand("help", "Xem trợ giúp"),
            BotCommand("crisis", "Tài nguyên khủng hoảng"),
            BotCommand("story", "Nhận câu chuyện trị liệu"),
            BotCommand("exercise", "Nhận bài tập thư giãn"),
            BotCommand("summary", "Xem tóm tắt cuộc trò chuyện"),
        ]
        await app.bot.set_my_commands(commands)
        logger.info("✓ Bot commands đã được thiết lập")
    
    async def bot_shutdown(app):
        await shutdown()
    
    application.post_init = bot_startup
    application.post_shutdown = bot_shutdown
    
    logger.info("🚀 Bắt đầu Enhanced Telegram Bot...")
    logger.info("📡 Bot đang chạy với 8 layers nâng cao. Gửi tin nhắn trên Telegram!")
    
    # Bắt đầu bot với long polling
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("⏹️ Bot dừng lại bởi người dùng")
    except Exception as e:
        logger.error(f"❌ Lỗi: {e}", exc_info=True)
