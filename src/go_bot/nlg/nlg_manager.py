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
        ai4eu_search_api_call_action: label of the action that corresponds to ai4eu search api call
            (it must be present in your ``template_path`` file), during interaction
            it will be used to get ``'db_result'`` from ``database``. (TODO update)
        ai4eu_qa_api_call_action: label of the action that corresponds to ai4eu QA api call
            (it must be present in your ``template_path`` file), during interaction
            it will be used to get ``'db_result'`` from ``database``. (TODO update)
        debug: whether to display debug output.
    """
    # static members used for human readable dates
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    def __init__(self, template_path: Union[str, Path], template_type: str,
                 ai4eu_search_api_call_action: str, ai4eu_qa_api_call_action: str, debug=False):
        self.debug = debug
        if self.debug:
            log.debug(f"BEFORE {self.__class__.__name__} init(): "
                      f"template_path={template_path}, template_typ e={template_type}, "
                      f"ai4eu_search_api_call_action={ai4eu_search_api_call_action}, debug={debug}, "
                      f"ai4eu_qa_api_call_action={ai4eu_qa_api_call_action}, debug={debug}")

        template_path = expand_path(template_path)
        template_type = getattr(go_bot_templates, template_type)
        self.templates = go_bot_templates.Templates(template_type).load(template_path)

        self._ai4eu_search_api_call_id = -1
        if ai4eu_search_api_call_action is not None:
            self._ai4eu_search_api_call_id = self.templates.actions.index(ai4eu_search_api_call_action)

        self._ai4eu_qa_api_call_id = -1
        if ai4eu_qa_api_call_action is not None:
            self._ai4eu_qa_api_call_id = self.templates.actions.index(ai4eu_qa_api_call_action)

        if self.debug:
            log.debug(f"AFTER {self.__class__.__name__} init(): "
                      f"template_path={template_path}, template_type={template_type}, "
                      f"ai4eu_search_api_call_action={ai4eu_search_api_call_action}, debug={debug}, "
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

    def get_ai4eu_search_api_call_action_id(self) -> int:
        """
        Returns:
            an ID corresponding to the ai4eu search api call action
        """
        return self._ai4eu_search_api_call_id

    def get_ai4eu_qa_api_call_action_id(self) -> int:
        """
        Returns:
            an ID corresponding to the ai4eu faq api call action
        """
        return self._ai4eu_qa_api_call_id

    def decode_response(self,
                        utterance_batch_features: BatchDialoguesFeatures,
                        policy_prediction: PolicyPrediction,
                        tracker_slotfilled_state) -> str:
        # todo: docstring

        action_text = self._generate_slotfilled_text_for_action(policy_prediction.predicted_action_ix,
                                                                tracker_slotfilled_state)
        # in api calls replace unknown slots to "dontcare"
        # TODO Check that this is correct for ai4eu_qa_api_call
        if policy_prediction.predicted_action_ix == self._ai4eu_search_api_call_id \
                or policy_prediction.predicted_action_ix == self._ai4eu_qa_api_call_id:
            action_text = re.sub("#([A-Za-z]+)", "dontcare", action_text).lower()
        return action_text

    def _generate_slotfilled_text_for_action(self, action_id: int, slots: dict) -> str:
        """
        Generate text for the predicted speech action using the pattern provided for the action.
        The slotfilled state provides info to encapsulate to the pattern.

        Args:
            action_id: the id of action to generate text for.
            slots: the slots and their known values. usually received from dialogue state tracker.

        Returns:
            the text generated for the passed action id and slot values.
        """

        # We have some templates that we create on the fly (e.g., date and time)
        action = self.get_action(action_id)

        # Check the action and create responses appropriately
        if action == 'tell_time':
            now = datetime.utcnow()
            text = 'The time is ' + now.strftime('%H:%M:%S') + ' UTC'
        elif action == 'tell_date':
            now = datetime.now()
            text = 'Today is ' + self.days[now.weekday()] + now.strftime(', %d ') + self.months[now.month - 1] + now.strftime(' %Y')
        else:
            text = self.templates.templates[action_id].generate_text(slots)

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
