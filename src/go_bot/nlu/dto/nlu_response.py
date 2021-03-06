from typing import Any, Dict, Tuple, List, Union, Optional

from .nlu_response_interface import NLUResponseInterface
from .text_vectorization_response import TextVectorizationResponse


class NLUResponse(NLUResponseInterface):
    """
    Stores the go-bot NLU knowledge: extracted slots and intents info, embedding and bow vectors.
    """
    def __init__(self, slots, intents, intent, tokens):
        self.slots: Union[List[Tuple[str, Any]], Dict[str, Any]] = slots
        self.intents = intents
        self.intent = intent
        self.tokens = tokens
        self.tokens_vectorized: Optional[TextVectorizationResponse] = None

    def set_tokens_vectorized(self, tokens_vectorized):
        self.tokens_vectorized = tokens_vectorized
