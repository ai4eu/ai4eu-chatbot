# FOUNDATION OF RESEARCH AND TECHNOLOGY - HELLAS (FORTH-ICS)
#
# INFORMATION SYSTEMS LABORATORY (ISL)
#
# http://www.ics.forth.gr/isl
#
# LICENCE: TO BE ADDED
#
# Copyright 2021

# The AI4EU chatbot - FAQ module

# This module can use two models that have been trained over a number of
# FAQ pairs gathered from the AI4EU web page
# The training data are available at data/ai4eu-faq/train.json

# Both models are sentence-embedding based models and provide rather good performance.
# Better than other models like a tf-idf logistic classifier or a BERT fine-tuned model

# The first model is the distilbert-base-nli-stsb-mean-tokens
# See https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens
# The second one is the language agnostic LaBSE that can potentially support
# a multi-language FAQ chat-bot
# See https://huggingface.co/sentence-transformers/LaBSE
# Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Narveen Ari, Wei Wang.
# Language-agnostic BERT Sentence Embedding. July 2020

# author: Papadakos Panagiotis
# e-mail: papadako@ics.forth.gr

from deeppavlov.core.commands.infer import build_model
import numpy as np


class FAQ:
    """
    distilbert provides good results and its not so heavy
    LaBSE supports many languages. Have to discuss this with Eric if we will support many languages
    Probabilities of the LaBSE model are less
    """
    def __init__(self, config='distilbert'):
        # Holds the available models. Currently only two are available
        self.__configs = {
            'distilbert': '../config/faq/sentence-emb/distilbert-base-nli-stsb-mean-tokens.json',
            'LaBSE': '../config/faq/sentence-emb/LaBSE.json'
        }

        # Check that this is a valid model else use distilbert
        if self.__configs[config] is None:
            config = 'distilbert'

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
    By default it loads the distilbert sentence embedding
    :return : True if loaded, else False
    """
    def load_model(self):
        self.__model = build_model(self.__configs[self.__config])
        print('FAQ loaded model ' + self.__config)

        if self.__model is not None:
            self.__loaded = True
        else:
            self.__loaded = False

        return self.__loaded

    """
    A method that asks the model a query
    :return :  The answer and its probability   
    """
    def ask(self, query):
        # Check that model is loaded
        if self.__loaded is False:
            return 'ERROR! FAQ model not loaded!', 1.0

        # Ask the model the user query. Get the prediction label and its probability
        result = self.__model.compute([query], targets=['y_pred_labels', 'y_pred_probas'])

        # Get the answer and its probability
        answer = result[0][0]
        prob = np.amax(result[1])

        # Print the result
        print('FAQ: Answer: ' + answer + ' probability:' + str(prob))

        return answer, prob