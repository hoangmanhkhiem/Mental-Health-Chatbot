# 🧠 Enhanced Mental Health Chatbot - 8 Layers

Hệ thống chatbot tư vấn tâm lý nâng cao với 8 module cải tiến theo tài liệu `doc.md`.

## 📦 Cấu trúc Modules

```
backend/
├── layers/                          # 8 Enhancement Layers
│   ├── __init__.py                  # Package exports
│   ├── emotion_analyzer.py          # Layer 3: Nhận diện cảm xúc
│   ├── user_memory.py               # Layer 2: Cá nhân hóa
│   ├── conversational_layer.py      # Layer 1: Hội thoại tự nhiên
│   ├── storytelling_therapy.py      # Layer 4: Kể chuyện trị liệu
│   ├── rag_precision.py             # Layer 5: RAG nâng cao
│   ├── reasoning_layer.py           # Layer 6: Suy luận
│   ├── safety_ethics.py             # Layer 7: An toàn & Đạo đức
│   └── proactive_dialogue.py        # Layer 8: Hội thoại chủ động
├── enhanced_chatbot.py              # Chatbot tích hợp tất cả layers
├── server_enhanced.py               # Server với enhanced chatbot
└── ...
```

## 🎯 8 Enhancement Layers

### Layer 1: Conversational Layer
**File:** `layers/conversational_layer.py`

Biến nội dung RAG khô khan thành hội thoại tự nhiên, đồng cảm.

```python
from layers import ConversationalLayer

layer = ConversationalLayer(model=gemini_model)

# Xây dựng prompt tự nhiên
prompt = layer.build_natural_prompt(
    user_message="Em rất lo lắng",
    rag_context=["context1", "context2"],
    emotion_analysis=emotion_result,
    personalization_context=user_context
)

# Tạo lời chào phù hợp
greeting = layer.generate_greeting(
    is_return_user=True,
    time_of_day="morning",
    last_topic="lo âu công việc"
)
```

### Layer 2: Personalization Layer
**File:** `layers/user_memory.py`

Lưu trữ và sử dụng thông tin cá nhân để cá nhân hóa trải nghiệm.

```python
from layers import UserMemoryStore

store = UserMemoryStore()

# Lưu tâm trạng
store.record_mood(user_id, "lo âu", "cao")

# Thêm mục tiêu trị liệu
store.set_therapy_goal(user_id, "giảm stress", "Đang theo dõi")

# Lấy context cá nhân hóa
context = store.get_personalization_context(user_id)
```

### Layer 3: Emotional Understanding
**File:** `layers/emotion_analyzer.py`

Tự động nhận diện cảm xúc và mức độ từ tin nhắn.

```python
from layers import EmotionAnalyzer

analyzer = EmotionAnalyzer(model=gemini_model)

# Phân tích cảm xúc
result = analyzer.analyze("Em rất buồn và không biết phải làm sao")

print(result.primary_emotion)  # EmotionType.SAD
print(result.intensity)        # EmotionIntensity.HIGH
print(result.suggested_tone)   # "đồng cảm, ấm áp, thấu hiểu"

# Lấy hướng dẫn phản hồi
guidelines = analyzer.get_response_guidelines(result)
```

### Layer 4: Storytelling Therapy
**File:** `layers/storytelling_therapy.py`

Tạo câu chuyện ẩn dụ và bài tập trị liệu.

```python
from layers import StorytellingTherapy
from layers.storytelling_therapy import TherapyApproach

therapy = StorytellingTherapy(model=gemini_model)

# Tạo câu chuyện ẩn dụ
story = therapy.generate_story(
    issue="lo âu giao tiếp",
    context=rag_context,
    emotion="lo âu",
    approach=TherapyApproach.CBT
)

# Lấy bài tập trị liệu
exercise = therapy.create_therapeutic_exercise(
    approach=TherapyApproach.MINDFULNESS,
    issue="stress"
)

# Tạo bài hình dung
visualization = therapy.create_visualization("căng thẳng", "cao")
```

### Layer 5: RAG Precision Boost
**File:** `layers/rag_precision.py`

Nâng cao độ chính xác truy xuất với multi-query, hybrid search, và reranking.

```python
from layers import RAGPrecisionBoost

rag_boost = RAGPrecisionBoost(model=gemini_model, rag_system=rag)

# Multi-query expansion
queries = rag_boost.generate_multi_queries("lo âu giao tiếp")
# ["lo âu giao tiếp", "anxiety when speaking", "social anxiety", ...]

# Hybrid search (semantic + BM25)
results = rag_boost.hybrid_search(db, query, k=10)

# Full precision pipeline
results = rag_boost.retrieve_with_precision(
    db, query,
    use_multi_query=True,
    use_hybrid=True,
    use_rerank=True
)

# Lấy context với citations
context, citations = rag_boost.get_context_with_citations(results)
```

### Layer 6: Reasoning Layer
**File:** `layers/reasoning_layer.py`

Chain-of-Thought và Self-Refinement để cải thiện chất lượng.

```python
from layers import ReasoningLayer

reasoning = ReasoningLayer(model=gemini_model)

# Chain-of-Thought reasoning
result = reasoning.reason_with_cot(query, context, explicit=False)

# Self-refinement
refined = reasoning.self_refine(draft_response, query, context)

# Full reasoning pipeline
final = reasoning.generate_with_reasoning(
    query=query,
    context=context,
    contexts=multiple_contexts,
    use_cot=True,
    use_refinement=True,
    use_synthesis=True
)
```

### Layer 7: Safety & Ethics
**File:** `layers/safety_ethics.py`

Kiểm tra an toàn, phát hiện khủng hoảng, và thêm disclaimers.

```python
from layers import SafetyEthicsLayer

safety = SafetyEthicsLayer(model=gemini_model)

# Kiểm tra tin nhắn người dùng
user_check = safety.check_user_message("Em muốn tự tử")
if user_check.risk_level.value == "critical":
    response = safety.get_crisis_response()

# Kiểm tra phản hồi bot
bot_check = safety.check_bot_response(response, user_message)
if bot_check.requires_action:
    response = bot_check.modified_content

# Xử lý hoàn chỉnh
final_response, check = safety.process_response(response, user_message)
```

### Layer 8: Proactive Dialogue Engine
**File:** `layers/proactive_dialogue.py`

Chatbot chủ động hỏi thăm và theo dõi tiến triển.

```python
from layers import ProactiveDialogueEngine

proactive = ProactiveDialogueEngine(
    model=gemini_model,
    memory_store=memory_store,
    safety_layer=safety_layer
)

# Cập nhật trạng thái hội thoại
state = proactive.update_state(user_id, message, emotion, topic)

# Kiểm tra hành động chủ động
actions = proactive.should_be_proactive(user_id, message)
if actions:
    best_action = actions[0]
    if best_action.should_act:
        follow_up_question = best_action.question

# Tạo câu hỏi chủ động với LLM
question = proactive.generate_proactive_question(
    user_id, conversation_history, rag_context
)

# Kiểm tra users không hoạt động
inactive_users = proactive.check_inactive_users(hours_threshold=24)
```

## 🚀 Enhanced Chatbot

File `enhanced_chatbot.py` tích hợp tất cả 8 layers vào một pipeline hoàn chỉnh.

```python
from enhanced_chatbot import create_enhanced_chatbot

# Tạo chatbot
chatbot = create_enhanced_chatbot(
    google_api_key=api_key,
    rag_system=rag,
    db_session_factory=session_factory
)

# Tạo response với full pipeline
response = chatbot.generate_response(
    db=db,
    user_id=user_id,
    user_message="Em rất lo lắng về công việc",
    use_rag=True,
    use_storytelling=False,
    use_proactive=True
)

print(response.response)           # Phản hồi cuối cùng
print(response.emotion_detected)   # Cảm xúc phát hiện
print(response.is_crisis)          # Có phải khủng hoảng?
print(response.used_rag)           # Đã dùng RAG?
print(response.proactive_elements) # Yếu tố chủ động thêm vào

# Lấy lời chào
greeting = chatbot.generate_greeting(user_id)

# Lấy tin nhắn check-in
check_in = chatbot.get_check_in_message(user_id)

# Lấy bài tập trị liệu
exercise = chatbot.get_therapeutic_exercise(user_id, "stress")
```

## 🌐 API Endpoints (Server Enhanced)

### Endpoints mới:

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/api/enhanced-chat` | POST | Chat với full 8 layers |
| `/api/storytelling` | POST | Chat với storytelling mode |
| `/api/exercise/{id}` | GET | Lấy bài tập trị liệu |
| `/api/summary/{id}` | GET | Lấy tóm tắt hội thoại |
| `/api/check-in/{id}` | GET | Lấy tin nhắn check-in |
| `/api/greeting/{id}` | GET | Lấy lời chào phù hợp |

### Ví dụ request:

```bash
# Enhanced chat
curl -X POST http://localhost:8000/api/enhanced-chat \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Em rất lo lắng về công việc",
    "conversation_id": "abc-123",
    "use_storytelling": false,
    "use_proactive": true
  }'

# Storytelling mode
curl -X POST http://localhost:8000/api/storytelling \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Em sợ thất bại",
    "conversation_id": "abc-123"
  }'

# Lấy bài tập
curl http://localhost:8000/api/exercise/abc-123?issue=stress
```

## 🏃 Chạy Server

```bash
# Server gốc
python server.py

# Server nâng cao với 8 layers
python server_enhanced.py
```

## 📊 Pipeline Tổng Quát

```
User message or silence
        ↓
┌─────────────────────────────┐
│   1. Safety Check           │ → Crisis? → Crisis Response
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   2. Emotion Analyzer       │ → {emotion, intensity}
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   3. Memory Reader          │ → User profile, history
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   4. Conversation State     │ → State tracking
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   5. Proactive Rules        │ → Should ask proactively?
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   6. RAG Precision Boost    │ → Multi-query, hybrid, rerank
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   7. Reasoning Layer        │ → CoT + Self-refinement
│      or Storytelling        │ → Generate story/metaphor
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   8. Conversational Layer   │ → Naturalize response
└─────────────────────────────┘
        ↓
┌─────────────────────────────┐
│   9. Safety Post-Check      │ → Add disclaimers
└─────────────────────────────┘
        ↓
    Final Response
```

## 🔧 Cấu hình

```env
# Required
GOOGLE_API_KEY=your_gemini_api_key

# Optional - Twilio
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_WHATSAPP_NUMBER=...

# Optional - Database
DATABASE_URL=postgresql://...

# Optional - Server
PORT=8000
HOST=0.0.0.0
ENVIRONMENT=production
```

## 📝 Ghi chú

- Tất cả layers hoạt động độc lập và có thể được sử dụng riêng lẻ
- Enhanced chatbot tích hợp tất cả layers theo pipeline tối ưu
- Backward compatible với các endpoints cũ (`/api/webChat`, `/api/awe-chat`)
- Safety layer luôn được áp dụng trước và sau khi tạo response
