"""
PHẦN 4 – Storytelling Therapy Mode (Tạo lời khuyên dưới dạng câu chuyện)

Mục tiêu:
- Biến kiến thức RAG thành lời khuyên dễ tiếp nhận
- Sử dụng ẩn dụ, ví dụ, câu chuyện (CBT, ACT, Narrative Therapy…)
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TherapyApproach(Enum):
    """Các phương pháp trị liệu."""
    CBT = "cognitive_behavioral"  # Liệu pháp nhận thức hành vi
    ACT = "acceptance_commitment"  # Liệu pháp chấp nhận và cam kết
    NARRATIVE = "narrative"  # Liệu pháp kể chuyện
    MINDFULNESS = "mindfulness"  # Chánh niệm
    SOLUTION_FOCUSED = "solution_focused"  # Tập trung giải pháp
    POSITIVE_PSYCHOLOGY = "positive_psychology"  # Tâm lý học tích cực


class StoryType(Enum):
    """Các loại câu chuyện."""
    METAPHOR = "ẩn dụ"
    PARABLE = "ngụ ngôn"
    PERSONAL_EXAMPLE = "ví dụ cá nhân"
    CASE_STUDY = "tình huống thực tế"
    VISUALIZATION = "hình dung"


@dataclass
class TherapeuticStory:
    """Câu chuyện trị liệu."""
    story_type: StoryType
    approach: TherapyApproach
    title: str
    content: str
    moral: str
    reflection_questions: List[str]


class StorytellingTherapy:
    """
    Module tạo câu chuyện trị liệu.
    Biến kiến thức khô khan thành câu chuyện ẩn dụ giúp người dùng hiểu vấn đề.
    """
    
    # Prompt templates cho các phương pháp trị liệu
    THERAPY_STORY_PROMPTS = {
        TherapyApproach.CBT: """Bạn là chuyên gia CBT (Cognitive Behavioral Therapy).
Hãy tạo một câu chuyện ẩn dụ giúp người dùng hiểu về:
- Mối liên hệ giữa suy nghĩ, cảm xúc và hành vi
- Cách nhận diện suy nghĩ tự động tiêu cực
- Cách thách thức và thay đổi suy nghĩ

VẤN ĐỀ: {issue}
CONTEXT: {context}

Câu chuyện nên:
- Dễ hiểu và gần gũi
- Có nhân vật người dùng có thể đồng cảm
- Kết thúc với bài học tích cực
- Có câu hỏi suy ngẫm""",

        TherapyApproach.ACT: """Bạn là chuyên gia ACT (Acceptance and Commitment Therapy).
Hãy tạo một câu chuyện ẩn dụ về:
- Chấp nhận cảm xúc thay vì chống cự
- Sống theo giá trị cá nhân
- Cam kết hành động có ý nghĩa

VẤN ĐỀ: {issue}
CONTEXT: {context}

Sử dụng ẩn dụ quen thuộc trong ACT như:
- Chiếc xe buýt (bạn là tài xế, suy nghĩ là hành khách)
- Bầu trời và mây (bạn là bầu trời, suy nghĩ là mây trôi qua)
- Cát lún (càng vùng vẫy càng chìm)""",

        TherapyApproach.NARRATIVE: """Bạn là chuyên gia Narrative Therapy.
Hãy giúp người dùng "viết lại câu chuyện" của họ:
- Tách vấn đề khỏi con người ("Bạn không phải là vấn đề")
- Tìm những ngoại lệ tích cực
- Xây dựng câu chuyện mới empowering

VẤN ĐỀ: {issue}
CONTEXT: {context}

Sử dụng kỹ thuật:
- Externalization (tách biệt vấn đề)
- Finding unique outcomes
- Re-authoring the story""",

        TherapyApproach.MINDFULNESS: """Bạn là hướng dẫn viên mindfulness.
Tạo một bài hướng dẫn thiền/chánh niệm dưới dạng câu chuyện:
- Tập trung vào hiện tại
- Quan sát không phán xét
- Từ bi với bản thân

VẤN ĐỀ: {issue}
CONTEXT: {context}

Câu chuyện có thể bao gồm:
- Hành trình thiền định
- Quan sát thiên nhiên như ẩn dụ
- Thực hành thở""",

        TherapyApproach.POSITIVE_PSYCHOLOGY: """Bạn là chuyên gia Positive Psychology.
Tạo câu chuyện tập trung vào:
- Điểm mạnh và đức hạnh cá nhân
- Lòng biết ơn
- Dòng chảy (flow) và hạnh phúc
- Ý nghĩa cuộc sống

VẤN ĐỀ: {issue}
CONTEXT: {context}

Kết hợp các yếu tố:
- PERMA (Positive emotions, Engagement, Relationships, Meaning, Achievement)
- Character strengths
- Gratitude practices"""
    }

    # Thư viện ẩn dụ có sẵn
    METAPHOR_LIBRARY = {
        "anxiety": [
            {
                "title": "Con sóng và người lướt ván",
                "content": """Hãy tưởng tượng lo âu như những con sóng biển. Khi mới học lướt ván, 
ta thường sợ sóng và cố gắng tránh chúng. Nhưng người lướt ván giỏi biết rằng sóng sẽ đến 
và đi - họ không cố dừng sóng, mà học cách cưỡi trên chúng.

Lo âu cũng vậy - nó đến như những con sóng. Thay vì cố gắng ngăn cản, ta có thể học 
cách "cưỡi" trên nó, biết rằng nó sẽ qua đi như mọi con sóng.""",
                "reflection": ["Bạn có thể nhớ lần nào lo âu đến rồi lại đi không?"]
            },
            {
                "title": "Chiếc thuyền trong bão",
                "content": """Có một chiếc thuyền nhỏ gặp bão giữa biển. Người thủy thủ hoảng sợ 
chèo điên cuồng, nhưng thuyền cứ lắc lư mạnh hơn. Một người lão làng nhìn thấy và nói:
"Hãy thả neo, để thuyền tự ổn định."

Khi lo âu đến, đôi khi ta cũng vùng vẫy như chiếc thuyền kia. Nhưng nếu ta "thả neo" - 
tập trung vào hơi thở, cảm nhận cơ thể - ta sẽ ổn định lại.""",
                "reflection": ["Neo của bạn có thể là gì?"]
            }
        ],
        "stress": [
            {
                "title": "Ly nước và gánh nặng",
                "content": """Một vị thầy cầm ly nước hỏi học trò: "Ly này nặng bao nhiêu?"
Học trò đoán: "200-500 gram?"
Thầy nói: "Trọng lượng không quan trọng. Nếu tôi giữ 1 phút, không sao. 1 giờ, tay tôi đau. 
1 ngày, tay tôi tê liệt."

Stress cũng vậy. Nó không nặng hay nhẹ tự thân - quan trọng là ta giữ nó bao lâu.
Hãy học cách đặt ly xuống thỉnh thoảng.""",
                "reflection": ["Bạn đang giữ ly nước nào quá lâu?"]
            }
        ],
        "depression": [
            {
                "title": "Khu vườn mùa đông",
                "content": """Mùa đông, khu vườn trông như đã chết. Không hoa, không lá xanh. 
Nhưng người làm vườn biết rằng dưới lớp đất lạnh, rễ cây vẫn đang sống, 
tích trữ năng lượng cho mùa xuân.

Khi bạn cảm thấy trống rỗng như khu vườn mùa đông, hãy nhớ: 
bạn không chết - bạn đang nghỉ ngơi, tích lũy sức mạnh cho mùa xuân của mình.""",
                "reflection": ["Mùa xuân của bạn sẽ trông như thế nào?"]
            }
        ],
        "self_worth": [
            {
                "title": "Viên kim cương trong bùn",
                "content": """Một viên kim cương rơi xuống bùn. Nó vẫn là kim cương, 
dù bị bùn che phủ. Giá trị của nó không thay đổi vì hoàn cảnh.

Bạn cũng vậy. Dù đang ở trong "bùn" của cuộc sống - thất bại, đau khổ, 
mất mát - giá trị bẩm sinh của bạn không thay đổi. Bùn chỉ là tạm thời.""",
                "reflection": ["Viên kim cương của bạn là gì?"]
            }
        ],
        "change": [
            {
                "title": "Con sâu hóa bướm",
                "content": """Trong kén, con sâu không biết nó sẽ thành bướm. 
Nó chỉ biết đang bị mắc kẹt trong bóng tối, thân thể đang tan rã.
Nhưng đó chính là quá trình chuyển hóa.

Khi bạn đang trong giai đoạn khó khăn, có thể bạn đang trong "kén" của mình. 
Sự khó chịu này có thể là dấu hiệu của sự chuyển đổi lớn.""",
                "reflection": ["Bạn đang chuyển hóa thành phiên bản nào của mình?"]
            }
        ]
    }

    STORY_GENERATION_PROMPT = """Hãy chuyển hóa kiến thức dưới đây thành một câu chuyện ẩn dụ 
giúp người dùng hiểu vấn đề tâm lý của họ một cách nhẹ nhàng và sâu sắc.

KIẾN THỨC/CONTEXT:
{context}

VẤN ĐỀ CỦA NGƯỜI DÙNG:
{issue}

CẢM XÚC HIỆN TẠI:
{emotion}

YÊU CẦU:
1. Tạo câu chuyện ẩn dụ ngắn gọn (150-250 từ)
2. Có nhân vật hoặc hình ảnh gần gũi
3. Liên hệ trực tiếp với vấn đề người dùng
4. Kết thúc với bài học hoặc góc nhìn mới
5. Thêm 1-2 câu hỏi suy ngẫm

Định dạng:
🌿 [Tiêu đề câu chuyện]

[Nội dung câu chuyện]

💭 Câu hỏi suy ngẫm:
- [Câu hỏi 1]
- [Câu hỏi 2]"""

    def __init__(self, model=None):
        """
        Khởi tạo Storytelling Therapy.
        
        Args:
            model: Gemini model instance (optional)
        """
        self.model = model
        logger.info("✓ StorytellingTherapy initialized")

    def get_metaphor(self, topic: str) -> Optional[Dict]:
        """
        Lấy ẩn dụ có sẵn cho chủ đề.
        
        Args:
            topic: Chủ đề (anxiety, stress, depression, etc.)
            
        Returns:
            Dict với metaphor hoặc None
        """
        import random
        
        metaphors = self.METAPHOR_LIBRARY.get(topic.lower())
        if metaphors:
            return random.choice(metaphors)
        
        # Try to find related topic
        topic_lower = topic.lower()
        for key, value in self.METAPHOR_LIBRARY.items():
            if key in topic_lower or topic_lower in key:
                return random.choice(value)
        
        return None

    def generate_story(
        self,
        issue: str,
        context: str = "",
        emotion: str = "",
        approach: TherapyApproach = None
    ) -> str:
        """
        Tạo câu chuyện trị liệu.
        
        Args:
            issue: Vấn đề của người dùng
            context: Context từ RAG
            emotion: Cảm xúc hiện tại
            approach: Phương pháp trị liệu
            
        Returns:
            Câu chuyện trị liệu
        """
        # Thử lấy metaphor có sẵn trước
        metaphor = self.get_metaphor(issue)
        if metaphor and not self.model:
            return self._format_metaphor(metaphor)
        
        if not self.model:
            return self._default_story(issue)
        
        try:
            # Sử dụng approach-specific prompt nếu có
            if approach and approach in self.THERAPY_STORY_PROMPTS:
                prompt = self.THERAPY_STORY_PROMPTS[approach].format(
                    issue=issue,
                    context=context
                )
            else:
                prompt = self.STORY_GENERATION_PROMPT.format(
                    context=context,
                    issue=issue,
                    emotion=emotion
                )
            
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"❌ Story generation error: {e}")
            if metaphor:
                return self._format_metaphor(metaphor)
            return self._default_story(issue)

    def _format_metaphor(self, metaphor: Dict) -> str:
        """Format ẩn dụ có sẵn."""
        formatted = f"🌿 **{metaphor['title']}**\n\n"
        formatted += metaphor['content']
        formatted += "\n\n💭 **Câu hỏi suy ngẫm:**\n"
        for q in metaphor.get('reflection', []):
            formatted += f"- {q}\n"
        return formatted

    def _default_story(self, issue: str) -> str:
        """Câu chuyện mặc định khi không có LLM."""
        return f"""🌿 **Hành trình nhỏ**

Đôi khi cuộc sống như một con đường dài với nhiều khúc quanh. 
Mỗi thử thách - như vấn đề "{issue}" mà bạn đang gặp - 
là một khúc quanh trên đường.

Nhưng hãy nhớ: mỗi khúc quanh cũng mang đến góc nhìn mới. 
Và bạn không đi một mình - mình ở đây cùng bạn.

💭 **Câu hỏi suy ngẫm:**
- Khúc quanh này có thể dạy bạn điều gì?
- Ai có thể đồng hành cùng bạn trên đoạn đường này?"""

    def create_visualization(self, issue: str, emotion: str = "") -> str:
        """
        Tạo bài hướng dẫn visualization.
        
        Args:
            issue: Vấn đề
            emotion: Cảm xúc
            
        Returns:
            Hướng dẫn visualization
        """
        if not self.model:
            return self._default_visualization(issue)
        
        try:
            prompt = f"""Tạo một bài hướng dẫn visualization ngắn (hình dung) để giúp người dùng 
thư giãn và xử lý vấn đề của họ.

VẤN ĐỀ: {issue}
CẢM XÚC: {emotion}

Bài hướng dẫn nên:
- Bắt đầu với hướng dẫn thở
- Dẫn dắt đến một nơi an toàn
- Sử dụng hình ảnh để xử lý cảm xúc
- Kết thúc bình yên
- Khoảng 200 từ"""
            
            response = self.model.generate_content(prompt)
            return "🧘 **Bài tập hình dung**\n\n" + response.text.strip()
            
        except Exception as e:
            logger.error(f"❌ Visualization error: {e}")
            return self._default_visualization(issue)

    def _default_visualization(self, issue: str) -> str:
        """Visualization mặc định."""
        return """🧘 **Bài tập hình dung: Nơi an toàn**

Hãy nhắm mắt và hít thở sâu 3 lần...

Bây giờ, hãy tưởng tượng bạn đang ở một nơi hoàn toàn an toàn. 
Có thể là bãi biển, khu rừng, hoặc căn phòng ấm áp của bạn.

Cảm nhận sự bình yên lan tỏa trong cơ thể...
Mỗi hơi thở ra, bạn thả đi một chút căng thẳng...

Ở đây, bạn được an toàn. 
Ở đây, bạn được là chính mình.

Khi bạn sẵn sàng, hãy từ từ mở mắt, mang theo sự bình yên này."""

    def suggest_therapy_approach(self, issue: str, personality: str = None) -> TherapyApproach:
        """
        Đề xuất phương pháp trị liệu phù hợp.
        
        Args:
            issue: Vấn đề
            personality: Tính cách người dùng
            
        Returns:
            TherapyApproach phù hợp
        """
        issue_lower = issue.lower()
        
        # Mapping vấn đề -> approach
        if any(word in issue_lower for word in ["suy nghĩ", "tiêu cực", "lo lắng", "tự phê bình"]):
            return TherapyApproach.CBT
        
        if any(word in issue_lower for word in ["chấp nhận", "không thể thay đổi", "buông bỏ"]):
            return TherapyApproach.ACT
        
        if any(word in issue_lower for word in ["câu chuyện", "quá khứ", "bản sắc", "tôi là ai"]):
            return TherapyApproach.NARRATIVE
        
        if any(word in issue_lower for word in ["stress", "căng thẳng", "thư giãn", "bình tĩnh"]):
            return TherapyApproach.MINDFULNESS
        
        if any(word in issue_lower for word in ["mạnh", "tích cực", "biết ơn", "hạnh phúc"]):
            return TherapyApproach.POSITIVE_PSYCHOLOGY
        
        # Default
        return TherapyApproach.MINDFULNESS

    def create_therapeutic_exercise(
        self,
        approach: TherapyApproach,
        issue: str
    ) -> str:
        """
        Tạo bài tập trị liệu.
        
        Args:
            approach: Phương pháp trị liệu
            issue: Vấn đề
            
        Returns:
            Bài tập
        """
        exercises = {
            TherapyApproach.CBT: """📝 **Bài tập nhận diện suy nghĩ**

1. Viết ra suy nghĩ đang làm bạn khó chịu
2. Hỏi: "Bằng chứng ủng hộ suy nghĩ này là gì?"
3. Hỏi: "Bằng chứng chống lại là gì?"
4. Viết ra một suy nghĩ cân bằng hơn

Ví dụ:
- Suy nghĩ: "Tôi thất bại"
- Bằng chứng ủng hộ: "Dự án này không thành công"
- Bằng chứng chống lại: "Tôi đã hoàn thành nhiều việc khác"
- Cân bằng: "Một dự án không thành công không định nghĩa tôi"
""",
            TherapyApproach.ACT: """🎯 **Bài tập xác định giá trị**

1. Hãy nghĩ về những gì thực sự quan trọng với bạn
2. Chọn 3 giá trị cốt lõi (ví dụ: gia đình, sáng tạo, tự do...)
3. Đánh giá: bạn đang sống theo những giá trị này như thế nào? (1-10)
4. Một hành động nhỏ bạn có thể làm hôm nay để sống đúng giá trị hơn là gì?

💡 Nhớ rằng: Cảm xúc khó chịu có thể đến và đi, nhưng hành động theo giá trị giúp bạn sống có ý nghĩa.
""",
            TherapyApproach.MINDFULNESS: """🧘 **Bài tập STOP**

Khi cảm thấy căng thẳng, hãy thử:

**S** - Stop (Dừng lại)
Tạm dừng những gì đang làm

**T** - Take a breath (Hít thở)
Hít sâu 3 lần, chú ý vào hơi thở

**O** - Observe (Quan sát)
Quan sát suy nghĩ, cảm xúc, cơ thể - không phán xét

**P** - Proceed (Tiếp tục)
Tiếp tục với sự nhận thức và ý định

Bạn có thể thử ngay bây giờ không?
""",
            TherapyApproach.POSITIVE_PSYCHOLOGY: """🌟 **Bài tập 3 điều tốt đẹp**

Mỗi tối trước khi ngủ:

1. Viết ra 3 điều tốt đẹp đã xảy ra hôm nay (dù nhỏ)
2. Ghi lại tại sao điều đó xảy ra
3. Cảm nhận lòng biết ơn trong vài giây

Ví dụ:
- "Được uống ly cà phê ngon → vì mình đã dậy sớm"
- "Bạn nhắn tin hỏi thăm → vì mình có người quan tâm"

Thực hành trong 1 tuần và cảm nhận sự thay đổi!
"""
        }
        
        return exercises.get(approach, exercises[TherapyApproach.MINDFULNESS])
