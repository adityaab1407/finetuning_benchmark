"""Approaches package for Financial Fine-Tuning Laboratory.

Contains implementations of all six benchmark approaches:
    1. Zero-shot prompting
    2. Few-shot prompting
    3. Chain-of-thought prompting
    4. RAG (Retrieval-Augmented Generation)
    5. SFT fine-tuned model (Mistral-7B + QLoRA)
    6. DPO-aligned model
"""

from approaches.base import ApproachResult, BaseApproach
from approaches.cot import ChainOfThoughtApproach
from approaches.few_shot import FewShotApproach
from approaches.rag import RAGApproach
from approaches.zero_shot import ZeroShotApproach

__all__ = [
    "ApproachResult",
    "BaseApproach",
    "ChainOfThoughtApproach",
    "FewShotApproach",
    "RAGApproach",
    "ZeroShotApproach",
]
