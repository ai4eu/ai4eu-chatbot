# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Updates for AI4EU

# FOUNDATION OF RESEARCH AND TECHNOLOGY - HELLAS (FORTH-ICS)
#
# INFORMATION SYSTEMS LABORATORY (ISL)
#
# http://www.ics.forth.gr/isl
#
# LICENCE: TO BE ADDED
#
# Copyright 2021

# The AI4EU chatbot - FAQ module

# This module is a modified version of the deeppavlov gobot for AI4EU

from logging import getLogger
from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.nn_model import NNModel
from .nlg.dto.nlg_response_interface import NLGResponseInterface
from .nlu.dto.text_vectorization_response import TextVectorizationResponse
from .nlu.tokens_vectorizer import TokensVectorizer
from .dto.dataset_features import UtteranceDataEntry, DialogueDataEntry, \
    BatchDialoguesDataset, UtteranceFeatures, UtteranceTarget, BatchDialoguesFeatures
from .dto.shared_gobot_params import SharedGoBotParams
from .nlg.nlg_manager import NLGManagerInterface
from .nlu.nlu_manager import NLUManager
from .policy.policy_network import PolicyNetwork, PolicyNetworkParams
from .policy.dto.policy_prediction import PolicyPrediction
from .search_api.search_api import SearchAPI
from .tracker.chatbot_mode import ChatMode
from .tracker.featurized_tracker import FeaturizedTracker
from .tracker.dialogue_state_tracker import DialogueStateTracker, MultipleUserStateTrackersPool
from pathlib import Path

from ..qa.ChatBot_QA import ChatBot_QA

log = getLogger(__name__)


# todo logging
@register("ai4eu_go_bot")
class AI4EUGoalOrientedBot(NNModel):
    """
    The dialogue bot is based on  https://arxiv.org/abs/1702.03274, which
    introduces Hybrid Code Networks that combine an RNN with domain-specific
    knowledge and system action templates.

    The network handles dialogue policy management.
    Inputs features of an utterance and predicts label of a bot action
    (classification task).

    An LSTM with a dense layer for input features and a dense layer for it's output.
    Softmax is used as an output activation function.

    Todo:
        add docstring for trackers.

    Parameters:
        tokenizer: one of tokenizers from
            :doc:`deeppavlov.models.tokenizers </apiref/models/tokenizers>` module.
        tracker: dialogue state tracker from
            :doc:`deeppavlov.models.go_bot.tracker </apiref/models/go_bot>`.
        hidden_size: size of rnn hidden layer.
        dropout_rate: probability of weights dropping out.
        l2_reg_coef: l2 regularization weight (applied to input and output layer).
        dense_size: rnn input size.
        attention_mechanism: describes attention applied to embeddings of input tokens.

            * **type** – type of attention mechanism, possible values are ``'general'``, ``'bahdanau'``,
              ``'light_general'``, ``'light_bahdanau'``, ``'cs_general'`` and ``'cs_bahdanau'``.
            * **hidden_size** – attention hidden state size.
            * **max_num_tokens** – maximum number of input tokens.
            * **depth** – number of averages used in constrained attentions
              (``'cs_bahdanau'`` or ``'cs_general'``).
            * **action_as_key** – whether to use action from previous time step as key
              to attention.
            * **intent_as_key** – use utterance intents as attention key or not.
            * **projected_align** – whether to use output projection.
        network_parameters: dictionary with network parameters (for compatibility with release 0.1.1,
            deprecated in the future)

        word_vocab: vocabulary of input word tokens
            (:class:`~deeppavlov.core.data.simple_vocab.SimpleVocabulary` recommended).
        bow_embedder: instance of one-hot word encoder
            :class:`~deeppavlov.models.embedders.bow_embedder.BoWEmbedder`.
        embedder: one of embedders from
            :doc:`deeppavlov.models.embedders </apiref/models/embedders>` module.
        slot_filler: component that outputs slot values for a given utterance
            (:class:`~deeppavlov.models.slotfill.slotfill.DstcSlotFillingNetwork`
            recommended).
        intent_classifier: component that outputs intents probability
            distribution for a given utterance (
            :class:`~deeppavlov.models.classifiers.keras_classification_model.KerasClassificationModel`
            recommended).
        use_action_mask: if ``True``, network output will be applied with a mask
            over allowed actions.
        debug: whether to display debug output.
    """

    DEFAULT_USER_ID = 1
    POLICY_DIR_NAME = "policy"

    def __init__(self,
                 tokenizer: Component,
                 tracker: FeaturizedTracker,
                 nlg_manager: NLGManagerInterface,
                 save_path: str,
                 hidden_size: int = 128,
                 dropout_rate: float = 0.,
                 l2_reg_coef: float = 0.,
                 dense_size: int = None,
                 attention_mechanism: dict = None,
                 network_parameters: Optional[Dict[str, Any]] = None,
                 load_path: str = None,
                 word_vocab: Component = None,
                 bow_embedder: Component = None,
                 embedder: Component = None,
                 slot_filler: Component = None,
                 intent_classifier: Component = None,
                 use_action_mask: bool = False,
                 debug: bool = False,
                 **kwargs) -> None:
        self.use_action_mask = use_action_mask  # todo not supported actually
        super().__init__(save_path=save_path, load_path=load_path, **kwargs)

        self.debug = debug

        # The following could be parameters in the json configuration
        self._TOPK = 3   # Topk results for qa and search modules
        # Threshold for action probabilities - Have to fine tune this
        self._THRESHOLD = 0.01

        # AI4EU Initialize the ChatBot_QA -> One instance for all user sessions
        self._QA = ChatBot_QA()
        # This is responsible for making requests to the search API
        self._SAPI = SearchAPI()

        # Params of policy LSTM component
        policy_network_params = PolicyNetworkParams(hidden_size, dropout_rate, l2_reg_coef,
                                                    dense_size, attention_mechanism, network_parameters)

        # Natural Language Understanding of user input. Takes the embeddings, the slot filler and the intents
        self.nlu_manager = NLUManager(tokenizer, slot_filler, intent_classifier)  # todo move to separate pipeline unit
        # Natural Language Generation, responsible for generating the responses
        self.nlg_manager = nlg_manager
        # Data handler is responsible for vectorizing the input utterance as an embedding
        self.data_handler = TokensVectorizer(debug, word_vocab, bow_embedder, embedder)

        # Initialize the Dialogue State Tracker using data from gobot
        self.dialogue_state_tracker = DialogueStateTracker.from_gobot_params(tracker,
                                                                             self.nlg_manager,
                                                                             policy_network_params,
                                                                             self._QA,
                                                                             self._SAPI,
                                                                             self._TOPK)
        # todo make more abstract
        self.multiple_user_state_tracker = MultipleUserStateTrackersPool(base_tracker=self.dialogue_state_tracker)

        # tokens embedding dimensions
        tokens_dims = self.data_handler.get_dims()
        print('==> AI4EU Token dims: ', tokens_dims.__dict__)

        # Set feature params
        features_params = SharedGoBotParams.from_configured(self.nlg_manager,
                                                            self.nlu_manager,
                                                            self.dialogue_state_tracker)
        print('==> AI4EU : Number of actions', features_params.num_actions)
        print('==> AI4EU : Number of intents', features_params.num_intents)
        print('==> AI4EU : Number of tracker features', features_params.num_tracker_features)

        policy_save_path = Path(save_path, self.POLICY_DIR_NAME)
        policy_load_path = Path(load_path, self.POLICY_DIR_NAME)

        # Initialize policy network
        self.policy = PolicyNetwork(policy_network_params, tokens_dims, features_params,
                                    policy_load_path, policy_save_path, **kwargs)

        self.dialogues_cached_features = dict()

        self.reset()

    def prepare_dialogues_batches_training_data(self,
                                                batch_dialogues_utterances_contexts_info: List[List[dict]],
                                                batch_dialogues_utterances_responses_info: List[
                                                    List[dict]]) -> BatchDialoguesDataset:
        """
        Parse the passed dialogue information to the dialogue information object.

        Args:
            batch_dialogues_utterances_contexts_info: the dictionary containing
                                                      the dialogue utterances training information
            batch_dialogues_utterances_responses_info: the dictionary containing
                                                       the dialogue utterances responses training information

        Returns:
            the dialogue data object containing the numpy-vectorized features and target extracted
            from the utterance data

        """
        # todo naming, docs, comments
        max_dialogue_length = max(len(dialogue_info_entry)
                                  for dialogue_info_entry in batch_dialogues_utterances_contexts_info)  # for padding

        batch_dialogues_dataset = BatchDialoguesDataset(max_dialogue_length)
        for dialogue_utterances_info in zip(batch_dialogues_utterances_contexts_info,
                                            batch_dialogues_utterances_responses_info):
            dialogue_index_value = dialogue_utterances_info[0][0].get("dialogue_label")

            if dialogue_index_value and dialogue_index_value in self.dialogues_cached_features.keys():
                dialogue_training_data = self.dialogues_cached_features[dialogue_index_value]
            else:
                dialogue_training_data = self.prepare_dialogue_training_data(*dialogue_utterances_info)
                if dialogue_index_value:
                    self.dialogues_cached_features[dialogue_index_value] = dialogue_training_data

            batch_dialogues_dataset.append(dialogue_training_data)

        return batch_dialogues_dataset

    def prepare_dialogue_training_data(self,
                                       dialogue_utterances_contexts_info: List[dict],
                                       dialogue_utterances_responses_info: List[dict]) -> DialogueDataEntry:
        """
        Parse the passed dialogue information to the dialogue information object.

        Args:
            dialogue_utterances_contexts_info: the dictionary containing the dialogue utterances training information
            dialogue_utterances_responses_info: the dictionary containing
                                                the dialogue utterances responses training information

        Returns:
            the dialogue data object containing the numpy-vectorized features and target extracted
            from the utterance data

        """
        dialogue_training_data = DialogueDataEntry()
        # we started to process new dialogue so resetting the dialogue state tracker.
        # simplification of this logic is planned; there is a todo
        self.dialogue_state_tracker.reset_state()
        for context, response in zip(dialogue_utterances_contexts_info, dialogue_utterances_responses_info):

            utterance_training_data = self.prepare_utterance_training_data(context, response)
            dialogue_training_data.append(utterance_training_data)

            # to correctly track the dialogue state
            # we inform the tracker with the ground truth response info
            # just like the tracker remembers the predicted response actions when real-time inference
            self.dialogue_state_tracker.update_previous_action(utterance_training_data.target.action_id)

            if self.debug:
                log.debug(f"True response = '{response['text']}'.")
                if utterance_training_data.features.action_mask[utterance_training_data.target.action_id] != 1.:
                    log.warning("True action forbidden by action mask.")
        return dialogue_training_data

    def prepare_utterance_training_data(self,
                                        utterance_context_info_dict: dict,
                                        utterance_response_info_dict: dict) -> UtteranceDataEntry:
        """
        Parse the passed utterance information to the utterance information object.

        Args:
            utterance_context_info_dict: the dictionary containing the utterance training information
            utterance_response_info_dict: the dictionary containing the utterance response training information

        Returns:
            the utterance data object containing the numpy-vectorized features and target extracted
            from the utterance data

        """
        # todo naming, docs, comments
        text = utterance_context_info_dict['text']

        # if there already were db lookups in this utterance
        # we inform the tracker with these lookups info
        # just like the tracker remembers the db interaction results when real-time inference
        # todo: not obvious logic
        # We are not using the ground truth since there is no ground truth
        # Search-API results are dynamic
        #self.dialogue_state_tracker.update_ground_truth_db_result_from_context(utterance_context_info_dict)

        utterance_features = self.extract_features_from_utterance_text(text, self.dialogue_state_tracker)

        action_id = self.nlg_manager.get_action_id(utterance_response_info_dict['act'])
        utterance_target = UtteranceTarget(action_id)

        utterance_data_entry = UtteranceDataEntry.from_features_and_target(utterance_features, utterance_target)
        return utterance_data_entry

    def extract_features_from_utterance_text(self, text, tracker, keep_tracker_state=False) -> UtteranceFeatures:
        """
        Extract ML features for the input text and the respective tracker.
        Features are aggregated from the
        * NLU;
        * text BOW-encoding&embedding;
        * tracker memory.

        Args:
            text: the text to infer to
            tracker: the tracker that tracks the dialogue from which the text is taken
            keep_tracker_state: if True, the tracker state will not be updated during the prediction.
                                Used to keep tracker's state intact when predicting the action
                                to perform right after the api call action is predicted and performed.

        Returns:
            the utterance features object containing the numpy-vectorized features extracted from the utterance
        """
        # todo comments

        nlu_response = self.nlu_manager.nlu(text)

        # region text BOW-encoding and embedding | todo: to nlu
        # todo move vectorization to NLU
        tokens_bow_encoded = self.data_handler.bow_encode_tokens(nlu_response.tokens)

        tokens_embeddings_padded = np.array([], dtype=np.float32)
        tokens_aggregated_embedding = np.array([], dtype=np.float32)
        if self.policy.has_attn():
            attn_window_size = self.policy.get_attn_window_size()
            # todo: this is ugly and caused by complicated nn configuration algorithm
            attn_config_token_dim = self.policy.get_attn_hyperparams().token_size
            tokens_embeddings_padded = self.data_handler.calc_tokens_embeddings(attn_window_size,
                                                                                attn_config_token_dim,
                                                                                nlu_response.tokens)
        else:
            tokens_aggregated_embedding = self.data_handler.calc_tokens_mean_embedding(nlu_response.tokens)
        nlu_response.set_tokens_vectorized(TextVectorizationResponse(
            tokens_bow_encoded,
            tokens_aggregated_embedding,
            tokens_embeddings_padded))
        # endregion text BOW-encoding and embedding | todo: to nlu

        # Change mode of bot based on specific intents
        # i.e. recognize when we are in QA, WEB, or ASSET mode
        if nlu_response.intent == 'qa_input':
            tracker.set_mode(ChatMode.QA)
        elif nlu_response.intent == 'web_resource_request':
            tracker.set_mode(ChatMode.WEB)
        elif nlu_response.intent == 'ai4eu_resource_request':
            tracker.set_mode(ChatMode.ASSET)

        # If we are just debugging do not update the state of tracker
        if nlu_response.intent == 'debug':
            keep_tracker_state = True

        if not keep_tracker_state:
            tracker.update_state(nlu_response, tracker.mode)

        tracker_knowledge = tracker.get_current_knowledge()

        digitized_policy_features = self.policy.digitize_features(nlu_response, tracker_knowledge)

        return UtteranceFeatures(nlu_response, tracker_knowledge, digitized_policy_features)

    def _infer(self, user_utterance_text: str, user_tracker: DialogueStateTracker,
               keep_tracker_state=False) -> Tuple[BatchDialoguesFeatures, PolicyPrediction, str]:
        """
        AI4EU Predict the action to perform in response to given text.

        Args:
            user_utterance_text: the user input text passed to the system
            user_tracker: the tracker that tracks the dialogue with the input-provided user
            keep_tracker_state: if True, the tracker state will not be updated during the prediction.
                                Used to keep tracker's state intact when predicting the action to perform right after
                                the api call action

        Returns:
            the features data object containing features fed to the model on inference and the model's prediction info and the intent
        """
        utterance_features = self.extract_features_from_utterance_text(user_utterance_text, user_tracker,
                                                                       keep_tracker_state)

        # Get also the intent in the response - intent is the 3 field in the NLUResponse response
        intent = utterance_features.nlu_response.intent

        utterance_data_entry = UtteranceDataEntry.from_features(utterance_features)

        # region pack an utterance to batch to further get features in batched form
        dialogue_data_entry = DialogueDataEntry()
        dialogue_data_entry.append(utterance_data_entry)
        # batch is single dialogue of 1 utterance => dialogue length = 1
        utterance_batch_data_entry = BatchDialoguesDataset(max_dialogue_length=1)
        utterance_batch_data_entry.append(dialogue_data_entry)
        # endregion pack an utterance to batch to further get features in batched form
        utterance_batch_features = utterance_batch_data_entry.features

        # as for RNNs: output, hidden_state < - RNN(output, hidden_state)
        hidden_cells_state, hidden_cells_output = user_tracker.network_state[0], user_tracker.network_state[1]

        policy_prediction = self.policy(utterance_batch_features,
                                        hidden_cells_state,
                                        hidden_cells_output,
                                        prob=True)

        # Set the vanilla utterance features to policy prediction
        # We need them for debugging reasons
        policy_prediction.set_utterance_features(utterance_features.nlu_response)

        return utterance_batch_features, policy_prediction, intent

    def __call__(self, batch: Union[List[List[dict]], List[str]],
                 user_ids: Optional[List] = None) -> Union[List[NLGResponseInterface],
                                                           List[List[NLGResponseInterface]]]:
        if isinstance(batch[0], list):
            # batch is a list of *completed* dialogues, infer on them to calculate metrics
            # user ids are ignored here: the single tracker is used and is reset after each dialogue inference
            # todo unify tracking: no need to distinguish tracking strategies on dialogues and realtime
            res = []
            for dialogue in batch:
                dialogue: List[dict]
                res.append(self._calc_inferences_for_dialogue(dialogue))
        else:
            # batch is a list of utterances possibly came from different users: real-time inference
            res = []
            if not user_ids:
                user_ids = [self.DEFAULT_USER_ID] * len(batch)
            for user_id, user_text in zip(user_ids, batch):
                user_text: str
                res.append(self._realtime_infer(user_id, user_text))

        print("==> AI4EU Chatbot response: ", res, '\n')

        return res

    # Main logic for real-time inference
    # We have different logic for different actions, since some of them change the current focus of the state
    # or have very specific functionality like the reset action
    def _realtime_infer(self, user_id, user_text) -> List[NLGResponseInterface]:
        # realtime inference logic
        #
        # we have the pool of trackers, each one tracks the dialogue with its own user
        # (1 to 1 mapping: each user has his own tracker and vice versa)

        # Get the tracker for this user
        user_tracker = self.multiple_user_state_tracker.get_or_init_tracker(user_id)
        responses = []

        # Policy Making -> Predict new action

        # predict the action to perform (e.g. response something or call any of the QA or search APIs)
        utterance_batch_features, policy_prediction, intent = self._infer(user_text, user_tracker)

        # Hold probability
        prob = float(policy_prediction.probs[policy_prediction.predicted_action_ix])

        # predicted action label
        pred_label = self.nlg_manager.get_action(policy_prediction.predicted_action_ix)

        # used action label
        used_label = self.nlg_manager.get_action(policy_prediction.predicted_action_ix)

        # Print the predicted action
        print(f"Predicted action is = '{pred_label}'")

        # AI4EU We have to check the probability of the actions.
        # If they are too low then it is better to use the QA module
        print(f"Probability of predicted action = '{prob}'")
        if policy_prediction.probs[policy_prediction.predicted_action_ix] < self._THRESHOLD:
            print(f"Fall-back to QA since prob is = '{ policy_prediction.probs[policy_prediction.predicted_action_ix]}'")
            policy_prediction.predicted_action_ix = self.nlg_manager.get_ai4eu_qa_api_call_action_id()
            used_label = self.nlg_manager.get_action(policy_prediction.predicted_action_ix)

        print(f"Use action = '{ self.nlg_manager.get_action(policy_prediction.predicted_action_ix)}'")

        # Update the action and the state for the next utterance for all actions
        # But do not update things for debug

        # If the captured intent is either debug or reset we do not care about the predicted action
        # Just do what they have to do
        if intent != 'debug' and intent != 'reset':
            user_tracker.update_previous_action(policy_prediction.predicted_action_ix)
            user_tracker.network_state = policy_prediction.get_network_state()
        else:
            # Make sure that we use the debug or reset action based on the intent and ignore the predicted action
            if intent == 'debug':
                policy_prediction.predicted_action_ix = self.nlg_manager.get_action_id('debug')
            elif intent == 'reset':
                policy_prediction.predicted_action_ix = self.nlg_manager.get_action_id('reset')

        # AI4EU: If we need to make a call to the AI4EU web search API for web resources or ai-catalogue assets
        if policy_prediction.predicted_action_ix == self.nlg_manager.get_ai4eu_web_search_api_call_action_id()\
                or policy_prediction.predicted_action_ix == self.nlg_manager.get_ai4eu_asset_search_api_call_action_id():

            # 1) we perform the api call for a web resource or an ai4eu asset
            if policy_prediction.predicted_action_ix == self.nlg_manager.get_ai4eu_web_search_api_call_action_id():
                user_tracker.make_ai4eu_web_search_api_call(user_text)
            elif policy_prediction.predicted_action_ix == self.nlg_manager.get_ai4eu_asset_search_api_call_action_id():
                # we might have a case where the user does not give a query but just requests for a result
                # in this case just use the slot values as user text
                # in the future search-API will also support some kind of wildcard query '*' that returns all results
                if intent == 'anything' or intent == 'make_query':
                    # get all slots and concatenate their values
                    slots = user_tracker.get_state()
                    # get all values from the slots and send the query
                    # The search-API will fail for empty queries with a 500
                    # if there are no slots and the focus will be none
                    query = " ".join(slots.values())

                    # clean up the values of the slots
                    # since they are concatenated with _ or have AI as a suffix
                    # TODO Have to clean up
                    query = query.replace('_', ' ')
                    if query.endswith('AI'):
                        query = query[:-2]
                        query = query + ' AI'

                    # if query is not empty
                    if query.strip():
                        user_tracker.make_ai4eu_asset_search_api_call(query)
                else:
                    user_tracker.make_ai4eu_asset_search_api_call(user_text)

            # 2) we predict what to do next
            # For now just assume that we are going to describe the resource and we are not going to predict another action
            # utterance_batch_features, policy_prediction = self._infer(user_text, user_tracker,
            #                                                          keep_tracker_state=True)
            #user_tracker.update_previous_action(policy_prediction.predicted_action_ix)
            #user_tracker.network_state = policy_prediction.get_network_state()

            # Prepare the response
            resp = self.nlg_manager.decode_response(utterance_batch_features,
                                                    policy_prediction,
                                                    user_tracker,
                                                    False)
            responses.append((resp, float(prob), pred_label, used_label, float(1.0)))

        # AI4EU: If we need to make a call to the AI4EU QA API, just call the QA component
        # No need to generate a response from action templates
        elif policy_prediction.predicted_action_ix == self.nlg_manager.get_ai4eu_qa_api_call_action_id():
            # just perform the qa api call
            # either use the QA component or the KBQA
            candidates = user_tracker.make_ai4eu_qa_api_call(user_text)

            # Log response from QA
            print(f"True response = '{candidates}'.")

            # AI4EU Now get the response and go back. There is no need to make a new prediction
            # We just report the response of the QA module
            resp = candidates[0][0]

            qa_ans_prob = candidates[0][1]

            # Append the response
            responses.append((resp, float(prob), pred_label, used_label, float(qa_ans_prob)))

        # Reset state since the user asks for a reset
        elif policy_prediction.predicted_action_ix == self.nlg_manager.get_action_id('reset'):
            # Reset state
            user_tracker.reset_state()
            # Prepare the response
            resp = self.nlg_manager.decode_response(utterance_batch_features,
                                                    policy_prediction,
                                                    user_tracker,
                                                    False)
            responses.append((resp, float(prob), pred_label, used_label, float(1.0)))
        else:
            # Prepare the response
            resp = self.nlg_manager.decode_response(utterance_batch_features,
                                                    policy_prediction,
                                                    user_tracker,
                                                    False)
            responses.append((resp, float(prob), pred_label, used_label, float(1.0)))

        return responses

    def _calc_inferences_for_dialogue(self, contexts: List[dict]) -> List[NLGResponseInterface]:
        # infer on each dialogue utterance
        # e.g. to calculate inference score via comparing the inferred predictions with the ground truth utterance
        # todo we provide the tracker with both predicted and ground truth response actions info. is this ok?
        # todo (response to ^) this should be used only on internal evaluations
        # todo warning.
        res = []
        self.dialogue_state_tracker.reset_state()
        for context in contexts:
            # PP TODO What about reset and debug?
            if context.get('prev_resp_act') is not None:
                # if there already were responses to user
                # we inform the tracker with these responses info
                # just like the tracker remembers the predicted response actions when real-time inference
                previous_action_id = self.nlg_manager.get_action_id(context['prev_resp_act'])
                self.dialogue_state_tracker.update_previous_action(previous_action_id)

            # if there already were db lookups
            # we inform the tracker with these lookups info
            # just like the tracker remembers the db interaction results when real-time inference
            # PP. The original code uses the provided db data as ground-trouth
            # In our implementation we ignore them since they come from dynamic data
            # self.dialogue_state_tracker.update_ground_truth_db_result_from_context(context)

            # infer things
            context_text = context['text']

            utterance_batch_features, policy_prediction, intent = self._infer(context_text, self.dialogue_state_tracker)

            # If debug or reset ignore it as action
            if intent != 'debug' and intent != 'reset':
                self.dialogue_state_tracker.update_previous_action(policy_prediction.predicted_action_ix)  # see above todo
            else:
                # Make sure that we use the debug or reset action based on the intent and ignore the predicted action
                if intent == 'debug':
                    policy_prediction.predicted_action_ix = self.nlg_manager.get_action_id('debug')
                elif intent == 'reset':
                    policy_prediction.predicted_action_ix = self.nlg_manager.get_action_id('reset')

            # If the predicted action updates the state then we have to update the state also to include new data
            # like the size of the response, reset of the state, moving the focus, etc.
            # AI4EU: If we need to make a call to the AI4EU web search API for web resources or ai-catalogue assets
            if policy_prediction.predicted_action_ix == self.nlg_manager.get_ai4eu_web_search_api_call_action_id() \
                    or policy_prediction.predicted_action_ix == self.nlg_manager.get_ai4eu_asset_search_api_call_action_id():

                # we perform the api call for a web resource or a asset
                if policy_prediction.predicted_action_ix == self.nlg_manager.get_ai4eu_web_search_api_call_action_id():
                    self.dialogue_state_tracker.make_ai4eu_web_search_api_call(context_text)
                elif policy_prediction.predicted_action_ix == self.nlg_manager.get_ai4eu_asset_search_api_call_action_id():
                    self.dialogue_state_tracker.make_ai4eu_asset_search_api_call(context_text)
            elif policy_prediction.predicted_action_ix == self.nlg_manager.get_action_id('reset'):
                self.dialogue_state_tracker.get_next_search_item()
            elif policy_prediction.predicted_action_ix == self.nlg_manager.get_action_id('tell_next_in_focus'):
                # get the next element in the focus
                self.dialogue_state_tracker.get_next_search_item()
            elif policy_prediction.predicted_action_ix == self.nlg_manager.get_action_id('tell_first_in_focus'):
                # get the first element
                self.dialogue_state_tracker.get_first_search_item()
            elif policy_prediction.predicted_action_ix == self.nlg_manager.get_action_id('tell_second_in_focus'):
                # get the second element
                self.dialogue_state_tracker.get_second_search_item()
            elif policy_prediction.predicted_action_ix == self.nlg_manager.get_action_id('tell_previous_in_focus'):
                # get the previous element
                self.dialogue_state_tracker.get_previous_search_item()

            # Get the network state
            self.dialogue_state_tracker.network_state = policy_prediction.get_network_state()

            # Prepare the response
            resp = self.nlg_manager.decode_response(utterance_batch_features,
                                                    policy_prediction,
                                                    self.dialogue_state_tracker,
                                                    True)

            # PP Was not sure how to change NLGResponseInterface for training
            # So just use the response text in training to keep things as originally developed
            res.append(resp)
        return res

    def train_on_batch(self,
                       batch_dialogues_utterances_features: List[List[dict]],
                       batch_dialogues_utterances_targets: List[List[dict]]) -> dict:
        batch_dialogues_dataset = self.prepare_dialogues_batches_training_data(batch_dialogues_utterances_features,
                                                                               batch_dialogues_utterances_targets)
        return self.policy.train_on_batch(batch_dialogues_dataset.features,
                                          batch_dialogues_dataset.targets)

    def reset(self, user_id: Union[None, str, int] = None) -> None:
        # WARNING: this method is confusing. todo
        # the multiple_user_state_tracker is applicable only to the realtime inference scenario
        # so the tracker used to calculate metrics on dialogues is never reset by this method
        # (but that tracker usually is reset before each dialogue inference)
        self.multiple_user_state_tracker.reset(user_id)
        if self.debug:
            log.debug("Bot reset.")

    def load(self, *args, **kwargs) -> None:
        self.policy.load()
        super().load(*args, **kwargs)

    def save(self, *args, **kwargs) -> None:
        super().save(*args, **kwargs)
        self.policy.save()
