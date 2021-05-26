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

# This module can use two models that have been trained over a number of FAQ gathered from the AI4EU web page
# The training data are available at data/ai4eu-faq/train.json

# Both models are sentence-embedding models and provide rather good performance.
# The first one is the distilbert-base-nli-stsb-mean-tokens
# See https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens
# The second one is the language agnostic LaBSE that could potentially provide a multi-language FAQ chat-bot
# See https://huggingface.co/sentence-transformers/LaBSE
# Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Narveen Ari, Wei Wang. Language-agnostic BERT Sentence Embedding. July 2020

# author: Papadakos Panagiotis
# e-mail: papadako@ics.forth.gr

from deeppavlov.core.commands.infer import build_model
import numpy as np


class FAQ:
    def __init__(self, config='distilbert'):
        # distilbert provides good results and its not so heavy
        # LaBSE supports many languages -> Have to discuss this with Eric -> Also works ok
        self.__configs = {
            'distilbert': '../config/faq/sentence-emb/distilbert-base-nli-stsb-mean-tokens.json',
            'LaBSE': '../config/faq/sentence-emb/LaBSE.json'
        }

        # Check that this is a valid model else use distilbert
        if self.__configs[config] is None:
            config = 'distilbert'

        # Holds the config
        self.__config = config
        # Holds the build that we have created for this config
        self.__model = None
        # Knows if we have loaded the model
        self.__loaded = False

        # Now load the model
        self.load_model()

    # a class method that builds the model
    def load_model(self):
        self.__model = build_model(self.__configs[self.__config])
        print('FAQ loaded model ' + self.__config)

        if self.__model is not None:
            self.__loaded = True

    # a class method that queries the model
    def ask(self, query):
        # Ask the model the user query. Get the prediction label and its probability
        result = self.__model.compute([query], targets=['y_pred_labels', 'y_pred_probas'])

        # Get the label and its probability
        label = result[0][0]
        prob = np.amax(result[1])

        # Print the result
        print('label: ' + label + ' probability:' + str(prob))

        return label, prob

