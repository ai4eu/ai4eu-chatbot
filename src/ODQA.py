# FOUNDATION OF RESEARCH AND TECHNOLOGY - HELLAS (FORTH-ICS)
#
# INFORMATION SYSTEMS LABORATORY (ISL)
#
# http://www.ics.forth.gr/isl
#
# LICENCE: TO BE ADDED
#
# Copyright 2021

# The AI4EU chatbot - Open Domain Question Answering

# author: Papadakos Panagiotis
# e-mail: papadako@ics.forth.gr

from deeppavlov import configs, build_model
import numpy as np


class ODQA:
    def __init__(self):
        self.__loaded = False
        self.__model = None

        # Now load the model
        self.load_model()

    # A class method that builds the model answering Open Domain Question Answering (ODQA)
    # using an exact answer to any question in Wikipedia articles.
    # Thus, given only a question, the system outputs the best answer it can find.
    # The default ODQA implementation takes a batch of queries as input and returns the best answer.
    # This is an expensive model. Needs about 24 GB of RAM
    def load_model(self):

        self.__model = build_model('../config/odqa/en_odqa_infer_wiki.json')

        if self.__model is not None:
            self.__loaded = True
        else:
            self.__loaded = False

    # a class method that queries the model
    def ask(self, query):
        # Check that model is loaded
        if self.__loaded is False:
            return 'ERROR! ODQA model not loaded!', 1.0

        # Ask the model the user query. Get the prediction label and its probability
        result = self.__model.compute([query], targets=['answer', 'answer_score'])

        # Get the label and its probability
        ans = result[0][0]
        score = np.amax(result[1])

        # Print the result
        print('ans: ' + ans + ' score:' + str(score))

        return ans, score

