# FOUNDATION OF RESEARCH AND TECHNOLOGY - HELLAS (FORTH-ICS)
#
# INFORMATION SYSTEMS LABORATORY (ISL)
#
# http://www.ics.forth.gr/isl
#
# LICENCE: TO BE ADDED
#
# Copyright 2021

# The AI4EU chatbot - Knowledge Based Question Answering

# This model is based on the wikidata knowledge base
# Wikidata models knowledge about a given subject in terms of concepts,
# entities, properties and relationships and can be used for answering factoid questions.

# author: Panagiotis Papadakos
# e-mail: papadako@ics.forth.gr

from deeppavlov import configs, build_model
import numpy as np


class KBQA:
    """
    The KBQA model that uses the wikidata KB. Query templates and entities are predicted using the BERT model,
    which are then linked with entities in wikidata entries. BiGRU ranks candidate relations and BERT candidate paths=
    """
    def __init__(self):
        self.__loaded = False
        self.__model = None

        # Now load the model
        self.load_model()
    """
    A method that loads the model
    :return : True if loaded, else False
    """
    def load_model(self):

        self.__model = build_model('config/kbqa/kbqa_cq.json')

        if self.__model is not None:
            self.__loaded = True
        else:
            self.__loaded = False

    """
    A method that asks the model a query
    k is the top-k results
    :return :  The answer, its score and the model
    """
    def ask(self, query, k):
        # Check that model is loaded
        if self.__loaded is False:
            return 'ERROR! KBQA model not loaded!', 1.0

        # Ask the model the user query. Get the prediction label and its probability
        result = self.__model.compute([query], targets=['answers'])

        # Get the ans
        ans = result[0]

        # Print the result
        print('Query: ' + query)
        print('KBQA ans: ' + ans)

        results = []
        item = ans, str(1.0)
        results.append(item)
        return 'kbqa', results
