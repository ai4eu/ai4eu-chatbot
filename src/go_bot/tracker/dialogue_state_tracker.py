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

from logging import getLogger
from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple, Any

import numpy as np

from deeppavlov.core.models.component import Component
from ..nlg.nlg_manager import NLGManagerInterface
from ..policy.dto.policy_network_params import PolicyNetworkParams
from ..search_api.search_api_results import SearchAPIResults
from ..tracker.dto.dst_knowledge import DSTKnowledge
from ..tracker.featurized_tracker import FeaturizedTracker

from ...qa.ChatBot_QA import ChatBot_QA

from ..search_api.search_api import SearchAPI

log = getLogger(__name__)


class DialogueStateTracker(FeaturizedTracker):
    def get_current_knowledge(self) -> DSTKnowledge:
        state_features = self.get_features()
        context_features = self.calc_context_features()
        knowledge = DSTKnowledge(self.prev_action,
                                 state_features, context_features,
                                 self._ai4eu_web_search_api_call_id,
                                 self._ai4eu_asset_search_api_call_id,
                                 self._ai4eu_qa_api_call_id,
                                 self.n_actions,
                                 self.calc_action_mask())
        return knowledge

    def __init__(self,
                 slot_names,
                 n_actions: int,
                 _ai4eu_web_search_api_call_id: int,
                 _ai4eu_asset_search_api_call_id: int,
                 _ai4eu_qa_api_call_id: int,
                 hidden_size: int,
                 qa: ChatBot_QA,
                 sapi: SearchAPI,
                 topk: int = 3,
                 domain_yml_path: Optional[Union[str, Path]]=None,
                 stories_yml_path: Optional[Union[str, Path]]=None,
                 **kwargs) -> None:
        super().__init__(slot_names, domain_yml_path, stories_yml_path, **kwargs)
        self.hidden_size = hidden_size
        self.topk = topk
        self.sapi = sapi
        self.qa = qa
        self.n_actions = n_actions
        self._ai4eu_web_search_api_call_id = _ai4eu_web_search_api_call_id
        self._ai4eu_asset_search_api_call_id = _ai4eu_asset_search_api_call_id
        self._ai4eu_qa_api_call_id = _ai4eu_qa_api_call_id
        self.ffill_act_ids2req_slots_ids: Dict[int, List[int]] = dict()
        self.ffill_act_ids2aqd_slots_ids: Dict[int, List[int]] = dict()
        self.reset_state()

        # The following features are used for computing the context feature vector
        self.curr_search_item = None            # holds the current search item
        self.curr_search_items = None           # holds the current topk items of search API results
        self.curr_search_item_index = 0         # holds the index of current item (starts from 1)
        self.curr_search_item_slot_state = None # holds the state used for querying

        #PP I am not sure that I will use the previous query state
        self.prev_search_item = None            # holds the active search item of the previous results
        self.prev_search_items = None           # topk items of search API results
        self.prev_search_item_index = 0         # index of current item

    @staticmethod
    def from_gobot_params(parent_tracker: FeaturizedTracker,
                          nlg_manager: NLGManagerInterface,
                          policy_network_params: PolicyNetworkParams,
                          qa: ChatBot_QA,
                          sapi: SearchAPI,
                          topk: int):
        slot_names = parent_tracker.slot_names

        # region set formfilling info
        act2act_id = {a_text: nlg_manager.get_action_id(a_text) for a_text in nlg_manager.known_actions()}
        action_id2aqd_slots_ids, action_id2req_slots_ids = DialogueStateTracker.extract_required_acquired_slots_ids_mapping(
            act2act_id, slot_names, nlg_manager, parent_tracker)

        # todo why so ugly and duplicated in multiple users tracker
        dialogue_state_tracker = DialogueStateTracker(slot_names, nlg_manager.num_of_known_actions(),
                                                      nlg_manager.get_ai4eu_web_search_api_call_action_id(),
                                                      nlg_manager.get_ai4eu_asset_search_api_call_action_id(),
                                                      nlg_manager.get_ai4eu_qa_api_call_action_id(),
                                                      policy_network_params.hidden_size,
                                                      qa,
                                                      sapi,
                                                      topk,
                                                      parent_tracker.domain_yml_path,
                                                      parent_tracker.stories_path)

        dialogue_state_tracker.ffill_act_ids2req_slots_ids = action_id2req_slots_ids
        dialogue_state_tracker.ffill_act_ids2aqd_slots_ids = action_id2aqd_slots_ids

        # endregion set formfilling info
        return dialogue_state_tracker

    @staticmethod
    def extract_required_acquired_slots_ids_mapping(act2act_id: Dict,
                                                    slot_names: List,
                                                    nlg_manager: NLGManagerInterface,
                                                    parent_tracker: FeaturizedTracker) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        get the required and acquired slots information for each known action in the 1-Hot Encoding form
        Args:
            act2act_id: the mapping of actions onto their ids
            slot_names: the names of slots known to the tracker
            nlg_manager: the NLG manager used in system
            parent_tracker: the tracker to take required and acquired slots information from

        Returns:
            the dicts providing np.array masks of required and acquired slots for each known action
        """
        action_id2aqd_slots_ids = dict()    # aqd stands for acquired
        action_id2req_slots_ids = dict()    # req stands for required
        for act in nlg_manager.known_actions():
            act_id = act2act_id[act]

            action_id2req_slots_ids[act_id] = np.zeros(len(slot_names), dtype=np.float32)
            action_id2aqd_slots_ids[act_id] = np.zeros(len(slot_names), dtype=np.float32)

            if isinstance(act, tuple):
                acts = act
            else:
                acts = [act]

            for act in acts:
                for slot_name_i, slot_name in enumerate(parent_tracker.action_names2required_slots.get(act, [])):
                    slot_ix_in_tracker = slot_names.index(slot_name)
                    action_id2req_slots_ids[act_id][slot_ix_in_tracker] = 1.
                for slot_name_i, slot_name in enumerate(parent_tracker.action_names2acquired_slots.get(act, [])):
                    slot_ix_in_tracker = slot_names.index(slot_name)
                    action_id2aqd_slots_ids[act_id][slot_ix_in_tracker] = 1.
        return action_id2aqd_slots_ids, action_id2req_slots_ids

    """
    Reset state of dialogue state tracker
    """
    def reset_state(self):
        super().reset_state()
        self.curr_search_item = None
        self.curr_search_items = None
        self.curr_search_item_slot_state = None
        self.curr_search_item_index = 0

        self.prev_search_item = None
        self.prev_search_items = None
        self.prev_search_item_index = 0
        self.prev_action = np.zeros(self.n_actions, dtype=np.float32)
        self._reset_network_state()

    """
    Reset network state of LSTM
    """
    def _reset_network_state(self):
        self.network_state = (
            np.zeros([1, self.hidden_size], dtype=np.float32),
            np.zeros([1, self.hidden_size], dtype=np.float32)
        )

    def update_previous_action(self, prev_act_id: int) -> None:
        self.prev_action *= 0.
        self.prev_action[prev_act_id] = 1.

    # AI4EU : We are not using a ground-truth in the DSTC-2 templates based on the returned results from API calls
    # We use only per_item_action_accuracy for our training instead of per_item_dialogue_accuracy
    def update_ground_truth_db_result_from_context(self, context: Dict[str, Any]):
        #self.current_db_result = context.get('db_result', None)
        #self._update_db_result()
        return None

    """
    Update search-api results 
    """
    def _update_search_results(self, results) -> None:
        if results is not None:
            log.info(f"Made ai4eu_web_search_api_call  got {len(results)} results.")
            # Hold the previous state
            self.prev_search_items = self.curr_search_items
            self.prev_search_item = self.curr_search_item
            self.prev_search_item_index = self.curr_search_item_index
            # Update the new state
            self.curr_search_items = results
            self.curr_search_item_index = 0
            self.curr_search_item = None
            self.curr_search_item_slot_state = self.get_state()  # Hold also the state for this query
        else:
            log.warning("Something went wrong with the search API")

    """
    Make call to search-API for web resources
    Store the topk results to the state
    """
    def make_ai4eu_web_search_api_call(self, user_text: str) -> None:
        # If we have a search API - Make the call to the API and get the results
        if self.sapi is not None:
            response = self.sapi.web_query(user_text, self.topk)
            # Get the results and update state
            results = SearchAPIResults.get_items(response)
            self._update_search_results(results)
            log.info(f"==> AI4EU Made ai4eu_web_search_api_call got {len(self.curr_search_items)} results.")
            # Activate the first item
            self.get_first_search_item()
        else:
            log.warning("No Search-API defined")


    """
    Make call to search-API for assets using the slot values from the state
    Store the topk results to the state
    """
    def make_ai4eu_asset_search_api_call(self, user_text: str) -> None:
        # If we have a search API - Make the call to the API and get the results
        if self.sapi is not None:
            # Get the current slots
            slots = self.get_state()

            # slot keys currently used for the AI4EU gobot
            valid_slots = ['researchArea', 'assetType', 'technicalCategories', 'businessCategories']

            # filter slot keys with value equal to 'dontcare'
            # also remove unknown slot keys (for example, 'this' in dstc2 tracker)
            filtered_slots = {
                s: v for s, v in slots.items() if v != 'dontcare' and s in valid_slots
            }

            # Ask the API for the results for this query
            response = self.sapi.ai_catalogue_query(user_text, self.topk,
                                                    research_area=filtered_slots.get('researchArea'),
                                                    asset_type=filtered_slots.get('assetType'),
                                                    technical_categories=filtered_slots.get('technicalCategories'),
                                                    business_categories=filtered_slots.get('businessCategories'))
            # Get the results and update state
            results = SearchAPIResults.get_items(response)
            self._update_search_results(results)

            log.info(f"==> AI4EU Made ai4eu_web_search_api_call got {len(self.curr_search_items)} results.")

            # Activate the first item
            self.get_first_search_item()
        else:
            log.warning("No Search-API defined")


    """
    Make call to QA-API 
            Args:
            user_text: the user input text passed to the system, which is currently considered the Question for QA

    We return the topk results along with their probabilities
    """
    @staticmethod
    def make_ai4eu_qa_api_call(self, user_text: str):
        # Ask the QA module the user request
        model, results = self.qa.ask(user_text, topk)

        log.info(f"Made ai4eu_qa_api_call with text {str}, got results {results} from model {model}.")

        # return the model and the results
        return results

    """
    Get the next search result
    The index of items starts from 1
    """
    def get_next_search_item(self):
        idx = self.curr_search_item_index + 1
        # If there are more results
        if self.curr_search_items is not None and len(self.curr_search_items) > idx:
            self.curr_search_item = SearchAPIResults.get_item_from_items(self.curr_search_items, idx)
            self.curr_search_item_index = idx
        else:
            self.curr_search_item = {}
            if self.curr_search_items is None:
                log.info(f"Can't get next search item. Asking for index {idx} from None curr_search_items ")
            else:
                log.info(f"Can't get next search item. Asking for index {idx}, "
                     f"while the length of the search results is  {len(self.curr_search_items)}")

    """
    Get the previous search result
    The index of items starts from 1
    """
    def get_previous_search_item(self):
        idx = self.curr_search_item_index - 1
        # If the focus was not the first element
        if self.curr_search_items is not None and idx > 0:
            self.curr_search_item = SearchAPIResults.get_item_from_items(self.curr_search_items, idx)
            self.curr_search_item_index = idx
        else:
            self.curr_search_item = {}
            if self.curr_search_items is None:
                log.info(f"Can't get previous search item. Asking for index {idx} from None curr_search_items ")
            else:
                log.info(f"Can't get previous search item. Asking for index {idx}, "
                     f"while the length of the search results is  {len(self.curr_search_items)}")

    """
    Get the first search result
    The index of items starts from 1
    """
    def get_first_search_item(self):
        idx = 1
        # If there are more results
        if self.curr_search_items is not None:
            self.curr_search_item = SearchAPIResults.get_item_from_items(self.curr_search_items, idx)
            self.curr_search_item_index = idx
        else:
            self.curr_search_item = {}
            if self.curr_search_items is None:
                log.info(f"Can't get the first search item. Asking for index {idx} from None curr_search_items ")
            else:
                log.info(f"Can't get first search item. Asking for index {idx}, "
                     f"while the length of the search results is  {len(self.curr_search_items)}")

    """
    Get the current search result
    The index of items starts from 1
    """
    def get_current_search_item(self):
        return self.curr_search_item

    """
    compute action mask
    """
    def calc_action_mask(self) -> np.ndarray:
        mask = np.ones(self.n_actions, dtype=np.float32)

        if np.any(self.prev_action):
            prev_act_id = np.argmax(self.prev_action)
            if prev_act_id == self._ai4eu_web_search_api_call_id or prev_act_id == self._ai4eu_asset_search_api_call_id:
                # Here we just mask the ai4eu search api call so that we can not make consecutive api requests
                # PP TODO Maybe there are cases that we might need an initial call for web resources and then for assets
                mask[prev_act_id] = 0.

        for act_id in range(self.n_actions):
            required_slots_mask = self.ffill_act_ids2req_slots_ids[act_id]
            acquired_slots_mask = self.ffill_act_ids2aqd_slots_ids[act_id]
            act_req_slots_fulfilled = np.equal((required_slots_mask * self._binary_features()), required_slots_mask)
            act_requirements_not_fulfilled = np.invert(act_req_slots_fulfilled)# if act_req_slots_fulfilled != [] else np.array([])
            ack_slot_is_already_known = np.equal((acquired_slots_mask * self._binary_features()), acquired_slots_mask)

            if any(act_requirements_not_fulfilled) or (all(ack_slot_is_already_known) and any(acquired_slots_mask)):
                mask[act_id] = 0.

        return mask

    """
    compute context features
    This is a feature vector based on the results of the search-api and the slots state
    Currently we only consider the latest results from the APIs
    We could also consider the previous one

    """
    def calc_context_features(self):
        dst_state = self.get_state()
        query_state = self.curr_search_item_slot_state
        result_matches_state = 0.
        # Check if the current slots match the query ones
        if query_state is not None:
            matching_items = dst_state.items()
            result_matches_state = all(v == query_state.get(s)
                                       for s, v in matching_items
                                       if v != 'dontcare') * 1.

        # Compute current value for focus index
        # If there are no results in focus it is zero
        # max results are topk
        focus_index = 0
        if self.curr_search_items is not None and len(self.curr_search_items) is not 0:
            focus_index = self.curr_search_item_index / len(self.curr_search_items)

        # Now compute the context features vector
        context_features = np.array([
            bool(self.curr_search_items) * 1.,          # current API results are not None
            (self.curr_search_items == {}) * 1.,        # current API results are empty
            (self.curr_search_item is None) * 1.,       # active API item is None
            bool(self.curr_search_item) * 1.,           # active API item is not None
            (self.curr_search_item == {}) * 1.,         # active API item is empty
            focus_index,                                # use also the current item index
            result_matches_state                        # if original query state matches current state
        ], dtype=np.float32)
        return context_features

    """
    We hold the results of the search API in self.curr_search_items
    Currently, we have no way to get from the search-API what slots the results satisfy
    So just return the slots and their values
    """
    def fill_current_state_with_searchAPI_results_slots_values(self) -> dict:
        slots = self.get_state()
    #    if self.db_result:
    #        for k, v in self.db_result.items():
    #            slots[k] = str(v)
        return slots

class MultipleUserStateTrackersPool(object):
    def __init__(self, base_tracker: DialogueStateTracker):
        self._ids_to_trackers = {}
        self.base_tracker = base_tracker

    def check_new_user(self, user_id: int) -> bool:
        return user_id in self._ids_to_trackers

    def get_user_tracker(self, user_id: int) -> DialogueStateTracker:
        if not self.check_new_user(user_id):
            raise RuntimeError(f"The user with {user_id} ID is not being tracked")

        tracker = self._ids_to_trackers[user_id]

        # TODO: understand why setting current_db_result to None is necessary
        #tracker.current_db_result = None
        return tracker

    def new_tracker(self):
        # todo deprecated and never used?
        tracker = DialogueStateTracker(self.base_tracker.slot_names, self.base_tracker.n_actions,
                                       self.base_tracker._ai4eu_web_search_api_call_id,
                                       self.base_tracker._ai4eu_asset_search_api_call_id,
                                       self.base_tracker._ai4eu_qa_api_call_id,
                                       self.base_tracker.hidden_size,
                                       self.base_tracker.qa,
                                       self.base_tracker.sapi,
                                       self.base_tracker.topk)
        return tracker

    def get_or_init_tracker(self, user_id: int):
        if not self.check_new_user(user_id):
            self.init_new_tracker(user_id, self.base_tracker)

        return self.get_user_tracker(user_id)

    def init_new_tracker(self, user_id: int, tracker_entity: DialogueStateTracker) -> None:
        # TODO: implement a better way to init a tracker
        # todo deprecated. The whole class should follow AbstractFactory or Pool pattern?
        tracker = DialogueStateTracker(
            tracker_entity.slot_names,
            tracker_entity.n_actions,
            tracker_entity._ai4eu_web_search_api_call_id,
            tracker_entity._ai4eu_asset_search_api_call_id,
            tracker_entity._ai4eu_qa_api_call_id,
            tracker_entity.hidden_size,
            tracker_entity.qa,
            tracker_entity.sapi,
            tracker_entity.topk,
            tracker_entity.domain_yml_path,
            tracker_entity.stories_path
        )
        tracker.ffill_act_ids2req_slots_ids = tracker_entity.ffill_act_ids2req_slots_ids
        tracker.ffill_act_ids2aqd_slots_ids = tracker_entity.ffill_act_ids2aqd_slots_ids

        self._ids_to_trackers[user_id] = tracker

    def reset(self, user_id: int = None) -> None:
        if user_id is not None and not self.check_new_user(user_id):
            raise RuntimeError(f"The user with {user_id} ID is not being tracked")

        if user_id is not None:
            self._ids_to_trackers[user_id].reset_state()
        else:
            self._ids_to_trackers.clear()
