"""
Enhanced Therapeutic Layers for Mental Health Chatbot
======================================================

This package contains 8 enhancement layers based on the doc.md specification:

1. Emotion Analyzer - Detects user emotions and intensity
2. User Memory Store - Stores user profiles and preferences
3. Conversational Layer - Transforms RAG content into natural conversation
4. Storytelling Therapy - Creates therapeutic narratives
5. RAG Precision Boost - Multi-query, hybrid retrieval, reranking
6. Reasoning Layer - Chain-of-thought and self-refinement
7. Safety & Ethics Layer - Content filtering and crisis handling
8. Proactive Dialogue Engine - Proactive questioning and follow-ups
"""

from .emotion_analyzer import EmotionAnalyzer
from .user_memory import UserMemoryStore
from .conversational_layer import ConversationalLayer
from .storytelling_therapy import StorytellingTherapy
from .rag_precision import RAGPrecisionBoost
from .reasoning_layer import ReasoningLayer
from .safety_ethics import SafetyEthicsLayer
from .proactive_dialogue import ProactiveDialogueEngine

__all__ = [
    'EmotionAnalyzer',
    'UserMemoryStore', 
    'ConversationalLayer',
    'StorytellingTherapy',
    'RAGPrecisionBoost',
    'ReasoningLayer',
    'SafetyEthicsLayer',
    'ProactiveDialogueEngine'
]
