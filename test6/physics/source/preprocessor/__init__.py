"""Self-contained photographed-question preprocessor used by test3."""

from .processor import ProcessResult, QuestionCard, QuestionImageProcessor
from .test5_adapter import Test5QuestionImageProcessor

__all__ = [
    "ProcessResult",
    "QuestionCard",
    "QuestionImageProcessor",
    "Test5QuestionImageProcessor",
]
