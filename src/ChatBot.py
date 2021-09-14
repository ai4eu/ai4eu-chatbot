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

import faq.FAQ as faq
import faq.KBQA as kbqa

class ChatBot:
    """
    Constructor of chatbot. Currently we have two models, the FAQ and the KBQA
    """
    def __init__(self):
        self.__faq = faq.FAQ()
        self.__kbqa = kbqa.KBQA()
        # Fine-tune this threshold especially for the FAQ model
        self.__THRESHOLD = 0.2

    """
    Ask the query to our models.
    First we ask the FAQ model. If the probability is less than the THRESHOLD,
    then we ask the KBQA model. If this also fails we return a default answer
    k is the top-k results
    :return :  The answer and its score 
    """
    def ask(self, query, k=3):
        # First ask the FAQ model
        model, results = self.__faq.ask(query, k)
        # Get the top result
        ans, score = results[0]

        # Add a threshold for our probabilities
        # If less than this then do something else
        if float(score) > self.__THRESHOLD and ans != 'INVALID':
            return model, results

        # Else we were not able to answer the question using the FAQ module
        # Now use the Knowledge Base Question Answering model
        model, results = self.__kbqa.ask(query, k)
        # Get the top result
        ans, score = results[0]
        # Currently KBQA just returns the top-ranked result
        # If empty then return rephrase
        if ans == 'Not Found':
            results = ['I did not understand! Can you please rephrase?'], 1.0
            return 'Default', results
        else:
            return model, results

    """
    Ask only the faq model and return the answer and the probability
    """
    def ask_faq(self, query, k):
        # ask the FAQ model
        model, results = self.__faq.ask(query, k)
        return model, results

    """
    Ask only the kbqa model and return the answer and the probability
    """
    def ask_kbqa(self, query, k):
        # ask the kbqa model
        model, results = self.__kbqa.ask(query, k)
        return model, results

