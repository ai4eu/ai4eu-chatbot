from typing import Tuple

import numpy as np

from ...nlu.dto.nlu_response import NLUResponse

class PolicyPrediction:
    """
    Used to store policy model predictions and hidden values.
    """
    def __init__(self, probs, prediction, hidden_outs, cell_state):
        self.probs = probs
        self.prediction = prediction
        self.hidden_outs = hidden_outs
        self.cell_state = cell_state
        self.predicted_action_ix = np.argmax(probs)
        self.utterance_features = None

    def get_network_state(self) -> Tuple:
        return self.cell_state, self.hidden_outs

    def set_utterance_features(self, features: NLUResponse):
        self.utterance_features = features

    def get_utterance_features(self) -> NLUResponse:
        return self.utterance_features
