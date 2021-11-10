from abc import ABCMeta, abstractmethod

from .dto.nlu_response_interface import NLUResponseInterface


class NLUManagerInterface(metaclass=ABCMeta):
    @abstractmethod
    def nlu(self, text) -> NLUResponseInterface:
        pass

    @abstractmethod
    def num_of_known_intents(self) -> int:
        """
        Returns:
            the number of intents known to the NLU module
        """
        pass
