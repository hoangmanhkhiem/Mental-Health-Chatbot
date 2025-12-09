"""
Bot Telegram AWE - Hỗ Trợ Sức Khỏe Tâm Thần

Xử lý tin nhắn Telegram với phản hồi hỗ trợ từ AI
Sử dụng thư viện python-telegram-bot với async/await

Tính năng:
- Xử lý lệnh /start, /help, /crisis
- Xử lý tin nhắn với ngữ cảnh RAG
- Phát hiện và xử lý khủng hoảng
- Theo dõi cuộc hội thoại
- Xử lý lỗi với fallback
"""

import os
import logging
import uuid
from datetime import datetime
from contextlib import asynccontextmanager

import nest_asyncio
from telegram import Update
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

# Chatbot + RAG
from chatbot import TherapeuticChatbot
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
chatbot = None
rag_system = None

# Kiểm tra token
if not TELEGRAM_BOT_TOKEN:
    logger.error("❌ TELEGRAM_BOT_TOKEN chưa được đặt")
    raise ValueError("Cần TELEGRAM_BOT_TOKEN trong .env")


async def startup():
    """Khởi tạo chatbot khi bot bắt đầu."""
    global chatbot, rag_system
    
    logger.info("=" * 70)
    logger.info("║    Bot Telegram AWE - Hỗ Trợ Sức Khỏe Tâm Thần         ║")
    logger.info("║              Hỗ Trợ Kỹ Thuật Số Thông Qua Telegram      ║")
    logger.info("=" * 70)
    
    # Khởi tạo RAG
    try:
        logger.info("📚 Khởi tạo hệ thống kiến thức (RAG)...")
        rag_system = TherapeuticRAG(google_api_key=GOOGLE_API_KEY)
        logger.info("✓ RAG khởi tạo thành công")
    except Exception as e:
        logger.warning(f"⚠️ Không khởi tạo RAG: {e}")
    
    # Khởi tạo Chatbot
    try:
        logger.info("🤖 Khởi tạo chatbot hỗ trợ...")
        chatbot = TherapeuticChatbot(
            google_api_key=GOOGLE_API_KEY,
            rag_system=rag_system
        )
        logger.info("✓ Chatbot khởi tạo thành công")
    except Exception as e:
        logger.error(f"✗ Khởi tạo chatbot thất bại: {e}")
        raise

    logger.info("=" * 70)
    logger.info("✅ Bot sẵn sàng!")
    logger.info("=" * 70 + "\n")


async def shutdown():
    """Dọn dẹp khi bot dừng."""
    logger.info("⏹️ Bot dừng lại")
    if SessionLocal:
        SessionLocal.close()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý lệnh /start."""
    user = update.effective_user
    
    welcome_message = (
        f"Xin chào {user.first_name}! 👋\n\n"
        "Tôi là trợ lý sức khỏe tâm thần từ AWE (Awareness-App). "
        "Tôi ở đây để cung cấp các cuộc trò chuyện hỗ trợ về sức khỏe kỹ thuật số, "
        "sức khỏe tâm thần và hạnh phúc cảm xúc.\n\n"
        "✨ **Tôi có thể giúp gì:**\n"
        "• Thời gian sử dụng màn hình và sức khỏe kỹ thuật số\n"
        "• Căng thẳng, lo âu và hỗ trợ cảm xúc\n"
        "• Kỹ thuật chánh niệm và hạnh phúc\n"
        "• Hỗ trợ sức khỏe chung\n\n"
        "📱 Chỉ cần gửi cho tôi một tin nhắn để bắt đầu cuộc trò chuyện.\n"
        "Nếu bạn gặp khủng hoảng, tôi sẽ cung cấp tài nguyên để hỗ trợ chuyên nghiệp.\n\n"
        "Lệnh: /help, /crisis"
    )
    
    await update.message.reply_text(welcome_message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý lệnh /help."""
    help_message = (
        "🆘 **Trợ Giúp & Hỗ Trợ**\n\n"
        "Tôi ở đây để cung cấp các cuộc trò chuyện hỗ trợ.\n\n"
        "**Lệnh:**\n"
        "/start - Bắt đầu cuộc trò chuyện\n"
        "/help - Hiển thị tin nhắn này\n"
        "/crisis - Nhận tài nguyên khủng hoảng\n\n"
        "**Chỉ cần gửi cho tôi một tin nhắn** - Tôi sẽ phản hồi bằng hỗ trợ liệu pháp.\n\n"
        "Nếu bạn đang gặp khủng hoảng sức khỏe tâm thần, vui lòng liên hệ với các chuyên gia:\n"
        "• 988 Đường dây sống sót và Khủng hoảng\n"
        "• Đường dây khủng hoảng: Gửi HOME đến 741741\n\n"
        "Sức khỏe của bạn rất quan trọng! ❤️"
    )
    
    await update.message.reply_text(help_message)


async def crisis_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý lệnh /crisis."""
    crisis_message = (
        "🆘 **Nếu Bạn Gặp Khủng Hoảng**\n\n"
        "Tôi biết bạn có thể cần hỗ trợ ngay lập tức. An toàn của bạn rất quan trọng.\n\n"
        "**Tài Nguyên Ngay Lập Tức:**\n\n"
        "🚨 Cấp cứu khẩn cấp: 113 – 114 – 115\n"
        "(113: Công an, 114: Cứu hoả, 115: Cấp cứu y tế)\n\n"
        "💬 Tư vấn tâm lý – Trung tâm 1088: Gọi 1900 1088\n"
        "(Có phí, hỗ trợ tâm lý – hoạt động 24/7)\n\n"
        "📞 Hội Tâm lý học Việt Nam: 1900 6162\n"
        "(Miễn phí, bảo mật, 24 giờ mỗi ngày)\n\n"
        "🏥 **Nếu gặp nguy hiểm tức thì:** Đến bệnh viện gần nhất hoặc tìm sự giúp đỡ ngay lập tức.\n\n"
        "Nhớ: Bạn không đơn độc. Những cô vấn được đào tạo sẵn sàng lắng nghe và hỗ trợ bạn. 💙"
    )
    
    await update.message.reply_text(crisis_message)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý tin nhắn văn bản."""
    user = update.effective_user
    chat_id = update.effective_chat.id
    user_message = update.message.text.strip()
    
    if not user_message:
        return
    
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
            logger.error("Chatbot chưa khởi tạo")
            return
        
        # Tạo phản hồi bằng chatbot
        try:
            response_dict = chatbot.generate_response(
                user_id=user.id,
                conversation_id=chat_id,
                user_message=user_message
            )
            
            response_text = response_dict.get("response", "")
            is_crisis = response_dict.get("is_crisis", False)
            
            logger.info(f"✓ Phản hồi được tạo (Khủng hoảng: {is_crisis}) - {response_text[:50]}...")
            
            # Gửi phản hồi
            await update.message.reply_text(response_text)
            
            # Ghi nhật ký cảnh báo nếu phát hiện khủng hoảng
            if is_crisis:
                logger.warning(f"⚠️ Phát hiện nội dung khủng hoảng từ người dùng {user.id}")
        
        except Exception as e:
            logger.error(f"✗ Lỗi xử lý tin nhắn: {e}", exc_info=True)
            
            # Phản hồi dự phòng
            fallback_response = (
                "Xin lỗi vì sự cố. Tôi gặp khó khăn trong việc xử lý tin nhắn của bạn lúc này.\n\n"
                "Nếu bạn gặp khủng hoảng, vui lòng liên hệ:\n"
                "• 988 Đường dây sống sót và Khủng hoảng\n"
                "• Đường dây khủng hoảng: Gửi HOME đến 741741\n\n"
                "Vui lòng thử lại hoặc liên hệ với đội hỗ trợ."
            )
            await update.message.reply_text(fallback_response)
    
    except Exception as e:
        logger.error(f"❌ Lỗi nghiêm trọng: {e}", exc_info=True)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Xử lý lỗi."""
    logger.error(f"Cập nhật gây ra lỗi: {context.error}", exc_info=True)


def main() -> None:
    """Bắt đầu bot Telegram."""
    
    # Tạo ứng dụng
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Thêm trình xử lý lệnh
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("crisis", crisis_command))
    
    # Thêm trình xử lý tin nhắn
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    
    # Thêm trình xử lý lỗi
    application.add_error_handler(error_handler)
    
    # Khởi tạo startup/shutdown tùy chỉnh
    async def bot_startup(app):
        await startup()
    
    async def bot_shutdown(app):
        await shutdown()
    
    application.post_init = bot_startup
    
    logger.info("🚀 Bắt đầu bot Telegram với long polling...")
    logger.info("📡 Bot đang chạy. Gửi tin nhắn cho bot của bạn trên Telegram")
    
    # Bắt đầu bot với long polling
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("⏹️ Bot dừng lại bởi người dùng")
    except Exception as e:
        logger.error(f"❌ Lỗi: {e}", exc_info=True)
