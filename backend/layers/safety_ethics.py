"""
PHẦN 7 – Safety & Ethics Layer

Mục tiêu:
- Tránh lời khuyên nguy hiểm
- Giữ ranh giới giữa trị liệu & chẩn đoán
- Xử lý trường hợp nguy cơ (self-harm)
- Hậu kiểm nội dung trước khi gửi
"""

import logging
import re
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Mức độ nguy cơ."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ContentType(Enum):
    """Loại nội dung cần kiểm tra."""
    USER_MESSAGE = "user_message"
    BOT_RESPONSE = "bot_response"
    PROACTIVE_QUESTION = "proactive_question"


@dataclass
class SafetyCheck:
    """Kết quả kiểm tra an toàn."""
    is_safe: bool
    risk_level: RiskLevel
    flags: List[str]
    requires_action: bool
    action_type: Optional[str]  # "crisis_response", "add_disclaimer", "block", "modify"
    modified_content: Optional[str]
    reason: str


class SafetyEthicsLayer:
    """
    Layer đảm bảo an toàn và đạo đức.
    Kiểm tra và xử lý nội dung nguy hiểm.
    """
    
    # Crisis keywords - mức độ cao nhất
    CRISIS_KEYWORDS = {
        "critical": [
            "tự tử", "muốn chết", "kết thúc cuộc sống", "không muốn sống",
            "tự làm đau", "cắt tay", "tự hại", "uống thuốc quá liều",
            "suicide", "kill myself", "end my life", "self-harm",
            "overdose", "want to die", "better off dead"
        ],
        "high": [
            "tuyệt vọng", "vô vọng", "không có lý do sống", "không ai cần",
            "sẽ biến mất", "mọi người sẽ tốt hơn khi không có tôi",
            "hopeless", "worthless", "burden to everyone",
            "no reason to live", "can't go on"
        ],
        "medium": [
            "không muốn thức dậy", "mệt mỏi với cuộc sống", "trầm cảm nặng",
            "không thể tiếp tục", "quá đau đớn",
            "tired of living", "can't take it anymore"
        ]
    }
    
    # Câu hỏi không nên hỏi chủ động
    PROHIBITED_PROACTIVE_QUESTIONS = [
        "bị lạm dụng", "lạm dụng tình dục", "bạo lực gia đình",
        "đang dùng thuốc tâm thần", "đã từng tự tử", "có ý định tự tử",
        "abuse", "trauma details", "medication names"
    ]
    
    # Nội dung cần disclaimer
    DISCLAIMER_TRIGGERS = [
        "chẩn đoán", "bệnh", "rối loạn", "thuốc", "điều trị y tế",
        "diagnosis", "disorder", "medication", "treatment"
    ]
    
    # Templates cho responses
    CRISIS_RESPONSE = """Mình rất lo lắng về những gì bạn đang chia sẻ. 
Cảm xúc của bạn là thật và quan trọng, và mình muốn bạn được an toàn.

🆘 **Vui lòng liên hệ ngay:**
- Đường dây nóng hỗ trợ tâm lý: 1800 599 920 (miễn phí, 24/7)
- Tổng đài sức khỏe tâm thần: 1900 0027
- Trong trường hợp khẩn cấp: Gọi 115

💚 Những người tư vấn chuyên nghiệp được đào tạo để hỗ trợ bạn qua giai đoạn này.

Bạn không đơn độc. Mình ở đây lắng nghe, nhưng những chuyên gia có thể hỗ trợ bạn tốt hơn."""

    SOFT_DISCLAIMER = """
💡 *Lưu ý: Mình là trợ lý AI hỗ trợ sức khỏe tinh thần, không thay thế tư vấn y tế chuyên nghiệp. 
Nếu bạn cần hỗ trợ chuyên sâu, hãy liên hệ chuyên gia tâm lý.*"""

    BOUNDARY_REMINDER = """
Mình không có khả năng đưa ra chẩn đoán y tế hay kê đơn thuốc. 
Để được đánh giá chính xác, bạn nên gặp bác sĩ hoặc chuyên gia tâm lý.
Mình có thể hỗ trợ bạn về cách đối phó và những bước tiếp theo."""

    def __init__(self, model=None):
        """
        Khởi tạo Safety & Ethics Layer.
        
        Args:
            model: Gemini model cho content review (optional)
        """
        self.model = model
        logger.info("✓ SafetyEthicsLayer initialized")
    
    def check_user_message(self, message: str) -> SafetyCheck:
        """
        Kiểm tra tin nhắn người dùng.
        
        Args:
            message: Tin nhắn người dùng
            
        Returns:
            SafetyCheck với kết quả
        """
        flags = []
        risk_level = RiskLevel.NONE
        action_type = None
        
        message_lower = message.lower()
        
        # Check critical keywords
        for keyword in self.CRISIS_KEYWORDS["critical"]:
            if keyword in message_lower:
                flags.append(f"Critical: {keyword}")
                risk_level = RiskLevel.CRITICAL
                action_type = "crisis_response"
                break
        
        # Check high risk if not critical
        if risk_level != RiskLevel.CRITICAL:
            for keyword in self.CRISIS_KEYWORDS["high"]:
                if keyword in message_lower:
                    flags.append(f"High risk: {keyword}")
                    risk_level = RiskLevel.HIGH
                    action_type = "crisis_response"
                    break
        
        # Check medium risk
        if risk_level == RiskLevel.NONE:
            for keyword in self.CRISIS_KEYWORDS["medium"]:
                if keyword in message_lower:
                    flags.append(f"Medium risk: {keyword}")
                    risk_level = RiskLevel.MEDIUM
                    action_type = "add_disclaimer"
                    break
        
        # Determine if safe
        is_safe = risk_level in [RiskLevel.NONE, RiskLevel.LOW, RiskLevel.MEDIUM]
        requires_action = risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        
        return SafetyCheck(
            is_safe=is_safe,
            risk_level=risk_level,
            flags=flags,
            requires_action=requires_action,
            action_type=action_type,
            modified_content=self.CRISIS_RESPONSE if action_type == "crisis_response" else None,
            reason="; ".join(flags) if flags else "No concerns detected"
        )
    
    def check_bot_response(self, response: str, user_message: str = "") -> SafetyCheck:
        """
        Hậu kiểm phản hồi của bot trước khi gửi.
        
        Args:
            response: Phản hồi của bot
            user_message: Tin nhắn người dùng (để context)
            
        Returns:
            SafetyCheck với kết quả và nội dung đã sửa nếu cần
        """
        flags = []
        modified = response
        needs_disclaimer = False
        
        response_lower = response.lower()
        
        # Check for inappropriate content
        inappropriate_patterns = [
            (r"bạn bị (bệnh|rối loạn|mắc)", "Potential diagnosis language"),
            (r"bạn nên uống thuốc", "Medication advice"),
            (r"bạn chắc chắn (có|bị)", "Definitive diagnosis"),
            (r"theo chẩn đoán", "Diagnostic language"),
        ]
        
        for pattern, reason in inappropriate_patterns:
            if re.search(pattern, response_lower):
                flags.append(reason)
        
        # Check if disclaimer needed
        for trigger in self.DISCLAIMER_TRIGGERS:
            if trigger in response_lower:
                needs_disclaimer = True
                break
        
        # Add disclaimer if needed
        if needs_disclaimer and self.SOFT_DISCLAIMER not in response:
            modified = response + "\n" + self.SOFT_DISCLAIMER
        
        # Check for harmful advice
        harmful_patterns = [
            r"không cần đi (bác sĩ|khám)",
            r"tự điều trị",
            r"bỏ thuốc",
            r"không cần lo",  # Dismissive
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, response_lower):
                flags.append(f"Potentially harmful: {pattern}")
        
        is_safe = len(flags) == 0
        
        return SafetyCheck(
            is_safe=is_safe,
            risk_level=RiskLevel.LOW if flags else RiskLevel.NONE,
            flags=flags,
            requires_action=needs_disclaimer or len(flags) > 0,
            action_type="modify" if modified != response else None,
            modified_content=modified if modified != response else None,
            reason="; ".join(flags) if flags else "Response is safe"
        )
    
    def check_proactive_question(self, question: str) -> SafetyCheck:
        """
        Kiểm tra câu hỏi chủ động trước khi gửi.
        
        Args:
            question: Câu hỏi định gửi
            
        Returns:
            SafetyCheck
        """
        flags = []
        question_lower = question.lower()
        
        # Check prohibited questions
        for prohibited in self.PROHIBITED_PROACTIVE_QUESTIONS:
            if prohibited in question_lower:
                flags.append(f"Prohibited question topic: {prohibited}")
        
        # Check for too personal questions
        personal_patterns = [
            r"bạn có bị .+ không\?",
            r"bạn đã từng .+ chưa\?",
            r"gia đình bạn có ai .+\?",
        ]
        
        for pattern in personal_patterns:
            if re.search(pattern, question_lower):
                flags.append("Too personal question")
        
        is_safe = len(flags) == 0
        
        return SafetyCheck(
            is_safe=is_safe,
            risk_level=RiskLevel.MEDIUM if flags else RiskLevel.NONE,
            flags=flags,
            requires_action=not is_safe,
            action_type="block" if flags else None,
            modified_content=None,
            reason="; ".join(flags) if flags else "Question is appropriate"
        )
    
    def add_appropriate_disclaimer(
        self,
        response: str,
        context_type: str = "general"
    ) -> str:
        """
        Thêm disclaimer phù hợp vào response.
        
        Args:
            response: Phản hồi gốc
            context_type: Loại ngữ cảnh (general, medical, diagnosis)
            
        Returns:
            Phản hồi với disclaimer
        """
        if context_type == "medical":
            disclaimer = self.BOUNDARY_REMINDER
        else:
            disclaimer = self.SOFT_DISCLAIMER
        
        # Don't add if already present
        if disclaimer in response or "Lưu ý:" in response:
            return response
        
        return response + "\n" + disclaimer
    
    def get_crisis_response(self, intensity: str = "high") -> str:
        """
        Lấy crisis response phù hợp.
        
        Args:
            intensity: Mức độ (medium, high, critical)
            
        Returns:
            Crisis response text
        """
        return self.CRISIS_RESPONSE
    
    def sanitize_response(self, response: str) -> str:
        """
        Làm sạch response, loại bỏ nội dung không phù hợp.
        
        Args:
            response: Phản hồi gốc
            
        Returns:
            Phản hồi đã làm sạch
        """
        sanitized = response
        
        # Remove any diagnostic language
        replacements = [
            (r"Bạn bị (bệnh |rối loạn )", "Có thể bạn đang trải qua "),
            (r"Bạn mắc ", "Bạn có thể đang gặp "),
            (r"Chẩn đoán của bạn là", "Một số dấu hiệu bạn mô tả có thể liên quan đến"),
        ]
        
        for pattern, replacement in replacements:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    async def review_with_llm(self, response: str, user_message: str) -> SafetyCheck:
        """
        Review nội dung sử dụng LLM (cho trường hợp phức tạp).
        
        Args:
            response: Phản hồi cần review
            user_message: Tin nhắn người dùng
            
        Returns:
            SafetyCheck
        """
        if not self.model:
            return self.check_bot_response(response, user_message)
        
        try:
            review_prompt = f"""Đánh giá phản hồi sau đây về mặt an toàn và đạo đức.

TIN NHẮN NGƯỜI DÙNG:
{user_message}

PHẢN HỒI CỦA BOT:
{response}

Kiểm tra:
1. Có đưa ra chẩn đoán y tế không?
2. Có khuyên dùng thuốc không?
3. Có nội dung có thể gây hại không?
4. Có phù hợp với cảm xúc người dùng không?
5. Có cần thêm disclaimer không?

Trả về JSON:
{{
    "is_safe": true/false,
    "concerns": ["concern1", "concern2"],
    "needs_disclaimer": true/false,
    "suggested_modifications": "..." (nếu cần)
}}"""
            
            result = self.model.generate_content(review_prompt)
            
            import json
            response_text = result.text.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            
            data = json.loads(response_text.strip())
            
            concerns = data.get("concerns", [])
            is_safe = data.get("is_safe", True)
            
            modified = response
            if data.get("needs_disclaimer"):
                modified = self.add_appropriate_disclaimer(response)
            if data.get("suggested_modifications"):
                modified = data["suggested_modifications"]
            
            return SafetyCheck(
                is_safe=is_safe,
                risk_level=RiskLevel.LOW if concerns else RiskLevel.NONE,
                flags=concerns,
                requires_action=not is_safe or data.get("needs_disclaimer", False),
                action_type="modify" if modified != response else None,
                modified_content=modified if modified != response else None,
                reason="; ".join(concerns) if concerns else "LLM review passed"
            )
            
        except Exception as e:
            logger.error(f"❌ LLM review error: {e}")
            return self.check_bot_response(response, user_message)
    
    def process_response(
        self,
        response: str,
        user_message: str,
        use_llm_review: bool = False
    ) -> Tuple[str, SafetyCheck]:
        """
        Xử lý hoàn chỉnh một response.
        
        Args:
            response: Phản hồi gốc
            user_message: Tin nhắn người dùng
            use_llm_review: Sử dụng LLM review không
            
        Returns:
            (processed_response, safety_check)
        """
        # Step 1: Basic check
        check = self.check_bot_response(response, user_message)
        
        # Step 2: Sanitize
        sanitized = self.sanitize_response(response)
        
        # Step 3: Apply modifications if needed
        final_response = sanitized
        if check.modified_content:
            final_response = check.modified_content
        
        # Step 4: Final check
        final_check = self.check_bot_response(final_response, user_message)
        
        return final_response, final_check


# Emergency resources by region
EMERGENCY_RESOURCES = {
    "vietnam": {
        "hotline": "1800 599 920",
        "mental_health": "1900 0027",
        "emergency": "115",
        "description": "Đường dây hỗ trợ tâm lý miễn phí 24/7"
    },
    "international": {
        "hotline": "International Association for Suicide Prevention",
        "website": "https://www.iasp.info/resources/Crisis_Centres/",
        "description": "Find local crisis centers"
    }
}
