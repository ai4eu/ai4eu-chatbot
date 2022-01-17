from abc import ABCMeta
from typing import Tuple


class NLGObjectResponseInterface(metaclass=ABCMeta):
    def to_serializable_dict(self) -> dict:
        raise NotImplementedError(f"to_serializable_dict() not implemented in {self.__class__.__name__}")


# PP str is the response, float is the probability of the predicted action
# second str is the type of predicted action, third str is the used action (if thresholds do not hold)
# and second float is the probability of the QA module
NLGResponseInterface = Tuple[NLGObjectResponseInterface, str, float, str, str, float]
