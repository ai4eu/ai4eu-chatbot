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

# author: Papadakos Panagiotis
# e-mail: papadako@ics.forth.gr

from deeppavlov import configs, build_model
import numpy as np


class KBQA:
    def __init__(self):
        self.__loaded = False
        self.__model = None

        # Now load the model
        self.load_model()

    # The knowledge base is a comprehensive repository of information about a given domain or a number of domains,
    # reflects the ways we model knowledge about a given subject or subjects, in terms of concepts, entities, properties, and relationships,
    # enables us to use this structured knowledge where appropriate, e.g., answering factoid questions
    def load_model(self):

        self.__model = build_model('../config/kbqa/kbqa_cq.json')

        if self.__model is not None:
            self.__loaded = True
        else:
            self.__loaded = False

    # a class method that queries the model
    def ask(self, query):
        # Check that model is loaded
        if self.__loaded is False:
            return 'ERROR! KBQA model not loaded!', 1.0

        # Ask the model the user query. Get the prediction label and its probability
        result = self.__model.compute([query], targets=['answers'])

        # Get the ans
        ans = result[0]

        # Print the result
        print('ans: ' + ans)

        return ans

# Now test our chatbot
chatbot = KBQA()

q = chatbot.ask('Which is the capital of Greece?')
print(q)