# FOUNDATION OF RESEARCH AND TECHNOLOGY - HELLAS (FORTH-ICS)
#
# INFORMATION SYSTEMS LABORATORY (ISL)
#
# http://www.ics.forth.gr/isl
#
# LICENCE: TO BE ADDED
#
# Copyright 2021

# The AI4EU chatbot - ChatBot module

# Currently we exploit two models for our chatbot
# The first one is a sentence based embedding model for FAQ trained over FAQ question-answer pairs
# over the AI4EU website and general AI terminology
# If the user query is not answered from this module with a probability greater than __THRESHOLD
# we use a KBQA model to get an answer from external Knowledge Bases (Wikidata).
# If everything fails we return a default answer

# author: Papadakos Panagiotis
# e-mail: papadako@ics.forth.gr

import FAQ
import KBQA


class ChatBot:
    """
    Constructor of chatbot. Currently we have two models, the FAQ and the KBQA
    """
    def __init__(self):
        self.__faq = FAQ.FAQ()
        self.__kbqa = KBQA.KBQA()
        # Fine-tune this threshold especially for the FAQ model
        self.__THRESHOLD = 0.1

    """
    Ask the query to our models.
    First we ask the FAQ model. If the probability is less than the THRESHOLD,
    then we ask the KBQA model. If this also fails we return a default answer
    :return :  The answer and its score 
    """
    def ask(self, query):
        # First ask the FAQ model
        ans, score, model = self.__faq.ask(query)

        # Add a threshold for our probabilities
        # If less than this then do something else
        if score > self.__THRESHOLD and ans != 'INVALID':
            return ans, score, model

        # Else we were not able to answer the question using the FAQ module
        # Now use the Knowledge Base Question Answering model
        ans, score, model = self.__kbqa.ask(query)

        # Currently KBQA just returns the top-ranked result
        # If empty then return rephrase
        if ans == 'Not Found':
            return 'I did not understand! Can you please rephrase?', 1.0, 'Default'
        else:
            return ans, score, model

    """
    Ask only the faq model and return the answer and the probability
    """
    def ask_faq(self, query):
        # ask the FAQ model
        ans, score, model = self.__faq.ask(query)
        return ans, score, model

    """
    Ask only the kbqa model and return the answer and the probability
    """
    def ask_kbqa(self, query):
        # ask the kbqa model
        ans, score, model = self.__kbqa.ask(query)
        return ans, score, model

