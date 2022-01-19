# FOUNDATION OF RESEARCH AND TECHNOLOGY - HELLAS (FORTH-ICS)
#
# INFORMATION SYSTEMS LABORATORY (ISL)
#
# http://www.ics.forth.gr/isl
#
# LICENCE: TO BE ADDED
#
# Copyright 2021

# The AI4EU chatbot

# Based on the goal oriented bot developed in deeppavlov with an integrated FAQ and KBQA module
# Uses the Search-API for retrieving web resources and assets from the AI4EU project

# author: Panagiotis Papadakos
# e-mail: papadako@ics.forth.g

from deeppavlov.core.commands.infer import build_model

# Code for importing the ai4eu_go_bot class
from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)

from src.go_bot import ai4eu_go_bot

class AI4EUChatbot:

    def __init__(self, config='ai4eu-gobot'):
        # Holds the available models. Currently, we only have one model
        self.__configs = {
            'ai4eu-gobot': 'config/gobot/ai4eu-gobot.json',
        }

        if self.__configs[config] is None:
            config = 'ai4eu-gobot'

        # Holds the config
        self.__config = config
        # Holds the model
        self.__model = None
        # Knows if we have loaded the model
        self.__loaded = False

        # Now load the model
        self.load_model()

    """
    A method that loads the model given in the constructor
    :return : True if loaded, else False
    """

    def load_model(self):
        self.__model = build_model(self.__configs[self.__config])
        print('AI4EU ChatBot loaded model ' + self.__config)

        if self.__model is not None:
            self.__loaded = True
        else:
            self.__loaded = False

        return self.__loaded

    """
    A method that asks the model a query
    query is the query
    :return :   Returns a single result, containing the response, the probability of the predicted action,
     the predicted action, the action used in the response and the probability of the specific answer (useful for QA)
    """

    def ask(self, query):
        # Check that model is loaded
        if self.__loaded is False:
            return 'ERROR! AI4EU Goal-oriented ChatBot model not loaded!', 1.0

        # Ask the model the user query. Get the prediction label and the probabilities
        result = self.__model.compute([query], targets=['y_predicted'])

        # Get the answer and all its metadata
        # Make it an array of responses, where each response is an array with the response and all metadata
        #{"results":[["Yes, you can!",0.31225526332855225,"can_do","can_do",1.0]],"model":"ai4eu-gobot","service":"AI4EU Goal-oriented ChatBot"}
        cleaned = [x for xs in result[0] for x in xs]

        # overwrite the first element with the clean version
        result[0] = cleaned
        return self.__config, result
