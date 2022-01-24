import re
from logging import getLogger
from pathlib import Path
from typing import Union, List

from deeppavlov.core.commands.utils import expand_path
import deeppavlov.models.go_bot.nlg.templates.templates as go_bot_templates
from deeppavlov.core.common.registry import register
from ..dto.dataset_features import BatchDialoguesFeatures
from .nlg_manager_interface import NLGManagerInterface
from ..policy.dto.policy_prediction import PolicyPrediction
from ..search_api.dto.search_item_in_focus import SearchItemInFocus
from ..tracker.dialogue_state_tracker import DialogueStateTracker

import numpy as np

from datetime import datetime

log = getLogger(__name__)


# todo add the ability to configure nlg loglevel in config (now the setting is shared across all the GO-bot)
# todo add each method input-output logging when proper loglevel level specified


@register("gobot_nlg_manager")
class NLGManager(NLGManagerInterface):
    """
    NLGManager is a unit of the go-bot pipeline that handles the generation of text
    when the pattern is chosen among the known patterns and the named-entities-values-like knowledge is provided.
    (the whole go-bot pipeline is as follows: NLU, dialogue-state-tracking&policy-NN, NLG)

    Parameters:
        template_path: file with mapping between actions and text templates
            for response generation.
        template_type: type of used response templates in string format.
        ai4eu_web_search_api_call_action: label of the action that corresponds to ai4eu search api call
            (it must be present in your ``template_path`` file), during interaction
            it will be used to get the appropriate results from the web resources of the search API
        ai4eu_asset_search_api_call_action: label of the action that corresponds to ai4eu search api call
            (it must be present in your ``template_path`` file), during interaction
            it will be used to get the appropriate assets from the ai-catalogue of the search API
        ai4eu_qa_api_call_action: label of the action that corresponds to ai4eu QA api call
            (it must be present in your ``template_path`` file), during interaction
            it will be used to get the relevant answer from the QA module (either a domain specific or open domain answer)
        debug: whether to display debug output.
    """
    # static members used for human readable dates
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    def __init__(self, template_path: Union[str, Path], template_type: str,
                 ai4eu_web_search_api_call_action: str,
                 ai4eu_asset_search_api_call_action: str,
                 ai4eu_qa_api_call_action: str,
                 debug=False):
        self.debug = debug
        if self.debug:
            log.debug(f"BEFORE {self.__class__.__name__} init(): "
                      f"template_path={template_path}, template_type={template_type}, "
                      f"ai4eu_web_search_api_call_action={ai4eu_web_search_api_call_action}, debug={debug}, "
                      f"ai4eu_asset_search_api_call_action={ai4eu_asset_search_api_call_action}, debug={debug}, "
                      f"ai4eu_qa_api_call_action={ai4eu_qa_api_call_action}, debug={debug}")

        template_path = expand_path(template_path)
        template_type = getattr(go_bot_templates, template_type)
        self.templates = go_bot_templates.Templates(template_type).load(template_path)

        # These actions are API related and are given in the gobot config json file
        self._ai4eu_web_search_api_call_id = -1
        if ai4eu_web_search_api_call_action is not None:
            self._ai4eu_web_search_api_call_id = self.templates.actions.index(ai4eu_web_search_api_call_action)

        self._ai4eu_asset_search_api_call_id = -1
        if ai4eu_asset_search_api_call_action is not None:
            self._ai4eu_asset_search_api_call_id = self.templates.actions.index(ai4eu_asset_search_api_call_action)

        self._ai4eu_qa_api_call_id = -1
        if ai4eu_qa_api_call_action is not None:
            self._ai4eu_qa_api_call_id = self.templates.actions.index(ai4eu_qa_api_call_action)

        if self.debug:
            log.debug(f"AFTER {self.__class__.__name__} init(): "
                      f"template_path={template_path}, template_type={template_type}, "
                      f"ai4eu_web_search_api_call_action={ai4eu_web_search_api_call_action}, debug={debug}, "
                      f"ai4eu_asset_search_api_call_action={ai4eu_asset_search_api_call_action}, debug={debug}, "
                      f"ai4eu_qa_api_call_action={ai4eu_qa_api_call_action}, debug={debug}")

    def get_action_id(self, action_text: str) -> int:
        """
        Looks up for an ID relevant to the passed action text in the list of known actions and their ids.

        Args:
            action_text: the text for which an ID needs to be returned.
        Returns:
            an ID corresponding to the passed action text
        """
        return self.templates.actions.index(action_text)  # todo unhandled exception when not found

    def get_ai4eu_web_search_api_call_action_id(self) -> int:
        """
        Returns:
            an ID corresponding to the ai4eu web search api call action
        """
        return self._ai4eu_web_search_api_call_id

    def get_ai4eu_asset_search_api_call_action_id(self) -> int:
        """
        Returns:
            an ID corresponding to the ai4eu asset search api call action
        """
        return self._ai4eu_asset_search_api_call_id

    def get_ai4eu_qa_api_call_action_id(self) -> int:
        """
        Returns:
            an ID corresponding to the ai4eu faq api call action
        """
        return self._ai4eu_qa_api_call_id

    def decode_response(self,
                        utterance_batch_features: BatchDialoguesFeatures,
                        policy_prediction: PolicyPrediction,
                        dialogue_state_tracker,
                        training=False) -> str:
        # todo: docstring

        action_text = self._generate_slotfilled_text_for_action(policy_prediction,
                                                                dialogue_state_tracker,
                                                                training)
        # in api calls replace unknown slots to "dontcare"
        # This is only needed for the asset search call that uses the slots
        # Hand-written actions and logic for APIs / reset / next object
        # TODO: Probably no need for this (REMOVE IT)
        #if policy_prediction.predicted_action_ix == self._ai4eu_asset_search_api_call_id:
        #    action_text = re.sub("#([A-Za-z]+)", "dontcare", action_text).lower()

        return action_text

    def _generate_slotfilled_text_for_action(self,
                                             policy_prediction: PolicyPrediction,
                                             dialogue_state_tracker: DialogueStateTracker,
                                             training=False) -> str:
        """
        Generate text for the predicted speech action using the pattern provided for the action.
        We need the state tracker for getting the slotfilled state that provides info to encapsulate to the patterns
        and for getting the current focus

        Args:
            policy_prediction: related info for policy prediction
            dialogue_state_tracker: holds the current state including the slots and current search item

        Returns:
            the text generated for the passed action id and slot values.
        """
        # current action id
        action_id = policy_prediction.predicted_action_ix

        # We have some templates that we create on the fly (e.g., API calls, focus info, date and time, etc.)
        action = self.get_action(action_id)

        # Update current searchAPI result slots / Just return the current state slots
        # * calculate the slotfilled state:
        #   for each slot that is relevant to dialogue we fill this slot value if possible
        #   unfortunately we can not make an inverse query and get the slots for a specific result
        #   currently we are using AND semantics
        slots = dialogue_state_tracker.fill_current_state_with_searchAPI_results_slots_values()

        # We also need the current search item
        item_in_focus = dialogue_state_tracker.get_current_search_item()

        # Check the action and create responses appropriately
        # These actions are specific for our chatbot
        # If we are training we are just using the dummy template responses for things that are dynamic
        # Else we create the corresponding responses
        if not training:
            # Respond with current debugging vectors
            if action == 'debug':
                text = self.tell_debug(policy_prediction, dialogue_state_tracker)
            # tell the url of the resource
            elif action == 'tell_resource_url':
                text = self.tell_resource_url(item_in_focus)
            # tell the title of the resource
            elif action == 'tell_resource_title':
                text = self.tell_resource_title(item_in_focus)
            # tell the content of the resource
            elif action == 'tell_resource_content':
                text = self.tell_resource_content(item_in_focus)
            # tell the score of the resource
            elif action == 'tell_resource_score':
                text = self.tell_resource_score(item_in_focus)
            # tell the summary of the resource
            elif action == 'tell_resource_summary':
                text = self.tell_resource_summary(item_in_focus)
            # tell the keywords of the resource
            elif action == 'tell_resource_keywords':
                text = self.tell_resource_keywords(item_in_focus)
            # tell the number of objects in focus
            elif action == 'tell_num_of_objects_in_focus':
                # get the current items of the focus
                items = dialogue_state_tracker.curr_search_items
                text = self.tell_objects_in_focus(items)
            # describe item in focus
            elif action == 'tell_item_in_focus':
                text = self.describe_item(item_in_focus)
            # describe next item in focus
            elif action == 'tell_next_in_focus':
                # we change the item in focus in the state to the next one
                item_in_focus = dialogue_state_tracker.get_next_search_item()
                text = self.describe_item(item_in_focus)
            # describe first item in focus when we have the tell_first_in_focus_action
            # or whenever we have a search API call
            elif action == 'tell_first_in_focus' or action == 'ai4eu_web_search_api_call':
                # we change the item in focus to the first one
                item_in_focus = dialogue_state_tracker.get_first_search_item()
                text = self.describe_item(item_in_focus)
            # describe second item in focus
            elif action == 'tell_second_in_focus':
                # we change the item in focus to the second one
                item_in_focus = dialogue_state_tracker.get_second_search_item()
                text = self.describe_item(item_in_focus)
            # describe previous item in focus
            elif action == 'tell_previous_in_focus':
                # we change the item in focus to the previous one
                item_in_focus = dialogue_state_tracker.get_previous_search_item()
                text = self.describe_item(item_in_focus)
            elif action == 'rephrase':
                text = self.templates.templates[action_id].generate_text(slots)
            # Respond with current UTC time
            elif action == 'tell_time':
                return self.tell_time()
            # Respond with current date
            elif action == 'tell_date':
                return self.tell_date()
            else:
                # General case - Just use the template
                text = self.templates.templates[action_id].generate_text(slots)
        else:
            # General case - Just use the template
            text = self.templates.templates[action_id].generate_text(slots)

        print('==> AI4EU Predicted response: ', text)
        return text

    # Provide debugging state as response
    # We have to report the intent, the slots, the current action and the previous action with their probabilities
    # Along with the current focus state
    def tell_debug(self, policy_prediction: PolicyPrediction, dialogue_state_tracker: DialogueStateTracker):
        text = '\n'
        ### NLU DATA - predicted intent, probability, and slots
        nlu_response = policy_prediction.get_utterance_features()
        intents = nlu_response.intents
        # Get the max probability
        max_prob = intents[np.argmax(intents)]
        intent = nlu_response.intent
        nlu = 'Predicted Intent: ' + str(intent) + ' with probability ' + str(max_prob) + '\n'
        # Also add slot-values from NLU
        slots = nlu_response.slots
        nlu += 'Slots: ' + str(slots) + '\n'
        nlu += '\n'
        text += nlu

        ### CURRENT ACTION
        # Print the predicted action
        action = 'Predicted action: ' + self.get_action(policy_prediction.predicted_action_ix) + '\n'
        action += 'Predicted action probability: ' + str(policy_prediction.probs[policy_prediction.predicted_action_ix]) + '\n'
        action += '\n'
        text += action

        ### REGARDING THE FOCUS
        current_focus_len = 'Empty'
        if dialogue_state_tracker.curr_search_items is not None:
            current_focus_len = len(dialogue_state_tracker.curr_search_items)
        # Regarding the current item in focus
        current_item_title = 'Empty'
        if dialogue_state_tracker.curr_search_item is not None:
            # Get the title of the search item in focus
            current_item_title = dialogue_state_tracker.curr_search_item.get_title()

        focus = 'Current focus length: ' + current_focus_len + '\n'
        focus += 'Current item title: ' + current_item_title + '\n'
        focus += 'Current item index: ' + str(dialogue_state_tracker.curr_search_item_index) + '\n'
        focus += '\n'
        text += focus

        return text

    '''
    Tells the title of a resource
    '''
    def tell_resource_title(self, item: SearchItemInFocus) -> str:
        response = None

        if not item:
            response = 'There is no item in the current focus'
        elif not item.get_title():
            response = 'This resource has no title'
        else:
            response = 'The title of the resource is ' + item.get_title()

        return response

    '''
    Tells the url of a resource
    '''
    def tell_resource_url(self, item: SearchItemInFocus) -> str:
        response = None

        if not item:
            response = 'There is no item in the current focus'
        elif not item.get_url():
            response = 'This resource has no url'
        else:
            # we need to offer clickable urls
            response = item.get_url()

        return response

    '''
    Tells the content of a resource
    '''
    def tell_resource_content(self, item: SearchItemInFocus) -> str:
        response = None

        if not item:
            response = 'There is no item in the current focus'
        elif not item.get_content():
            response = 'This resource has empty content'
        else:
            # we need to offer clickable urls
            response = item.get_content()

        return response

    '''
    Tells the score of a resource
    '''
    def tell_resource_score(self, item: SearchItemInFocus) -> str:
        response = None

        if not item:
            response = 'There is no item in the current focus'
        elif not item.get_score():
            response = 'The API returned no score for this resource'
        else:
            response = 'The score of this resource is ' + item.get_score()

        return response

    '''
    Tells the summary of a resource
    '''
    def tell_resource_summary(self, item: SearchItemInFocus) -> str:
        response = None

        if not item:
            response = 'There is no item in the current focus'
        elif not item.get_score():
            response = 'There is no summary for this resource'
        else:
            response = item.get_summary()

        return response

    '''
    Tells the keywords of a resource
    '''
    def tell_resource_keywords(self, item: SearchItemInFocus) -> str:
        response = None

        if not item:
            response = 'There is no item in the current focus'
        elif not item.get_keywords():
            response = 'There are no keywords associated with this resource'
        else:
            keywords = item.get_keywords()
            print(' The keywords are ', keywords)
            keywords_str = ' '.join(map(str, keywords))
            response = 'The relevant keywords, starting from the most important one are : ' + keywords_str

        return response

    '''
    Tells the number of items in focus of a resource
    '''
    def tell_objects_in_focus(self, items: [SearchItemInFocus]) -> str:
        response = None

        if not items:
            response = 'There are no items in the current focus'
        else:
            response = 'There are ' + len(items) + ' items in the current focus'

        return response

    '''
    Describes an item
    '''
    def describe_item(self, item: SearchItemInFocus) -> str:
        response = None

        if not item:
            response = 'There is no item in the current focus'
        else:
            item.print()
            response = 'You might be interested in ' + item.get_title() + '. Check it at: ' + item.get_url()

        return response

    # Tell the time
    def tell_time(self):
        now = datetime.utcnow()
        text = 'The time is ' + now.strftime('%H:%M:%S') + ' UTC'
        return text

    # Tell the date
    def tell_date(self):
        now = datetime.now()
        text = 'Today is ' + self.days[now.weekday()] + now.strftime(', %d ') + self.months[
            now.month - 1] + now.strftime(' %Y')
        return text

    def num_of_known_actions(self) -> int:
        """
        Returns:
            the number of actions known to the NLG module
        """
        return len(self.templates)

    def known_actions(self) -> List[str]:
        """
        Returns:
             the list of actions known to the NLG module
        """
        return self.templates.actions

    def get_action(self, action_id: int) -> str:
        """
        Returns:
             the action with id known to the NLG module
        """
        return self.templates.actions[action_id]
