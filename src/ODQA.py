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

# ODQA is the Open Domain Question Answering model
# It is very expensive in terms of space complexity and it is rather slow in inference
# Accuracy is rather low for real-life application

# author: Papadakos Panagiotis
# e-mail: papadako@ics.forth.gr

from deeppavlov import configs, build_model
import numpy as np


class ODQA:
    """
    Constructor of Open Domain Question Answer (ODQA)
    using an exact answer to any question in Wikipedia articles.
    Thus, given only a question, the system outputs the best answer it can find.
    The default ODQA implementation takes a batch of queries as input and returns the best answer.
    This is an expensive model. Needs about 24 GB of RAM
    Currently not used since it is rather expensive and not very accurate
    """
    def __init__(self):
        self.__loaded = False
        self.__model = None

        # Now load the model
        self.load_model()

    """
    A method that loads the model answering Open Domain Question Answering (ODQA)
    :return : True if loaded, else False
    """
    def load_model(self):

        self.__model = build_model('../config/odqa/en_odqa_infer_wiki.json')

        if self.__model is not None:
            self.__loaded = True
        else:
            self.__loaded = False

    """
    A method that asks the model a query
    :return :  The answer and its score   
    """
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