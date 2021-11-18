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

# This module various models that have been trained over a number of
# FAQ pairs gathered from the AI4EU web page
# The training data are available at data/ai4eu-faq/train.json

# All models are sentence-embedding based models and provide rather good performance.
# They offer better than other models like a tf-idf logistic classifier or a BERT fine-tuned model

# Used the best models from https://www.sbert.net/docs/pretrained_models.html

# The default model we are using is the all-mpnet-base-v2, which is the best one in many tasks
# although a bit slow
# https://huggingface.co/sentence-transformers/all-mpnet-base-v2

# The state-of-the-art paraphrase model mpnet (it also includes a multilingual model)
# https://huggingface.co/transformers/model_doc/mpnet.html
# https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
# https://arxiv.org/abs/2004.09297

# Also a multilingual mpnet version

# The faster microsoft/MiniLM-L12-H384-uncased (problem with dp 0.16.0)
# https://huggingface.co/microsoft/MiniLM-L12-H384-uncased

# distilbert-base-nli-stsb-mean-tokens (problem with dp 0.16.0)
# See https://huggingface.co/sentence-transformers/distilbert-base-nli-stsb-mean-tokens

# The language agnostic LaBSE that can potentially support a multi-language FAQ chat-bot
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

    def __init__(self, config='all-mpnet'):
        # Holds the available models. Currently only two are available
        self.__configs = {
            'all-mpnet': 'config/qa/sentence-emb/all-mpnet-base-v2.json',
            'multi-qa': 'config/qa/sentence-emb/multi-qa-mpnet-base-dot-v1.json',
            'MiniLM-12': 'config/qa/sentence-emb/MiniLM-L12-H384-uncased.json',
            'MiniLM-6': 'config/qa/sentence-emb/MiniLM-L6-H384-uncased.json',
            'distilbert': 'config/qa/sentence-emb/distilbert-base-nli-stsb-mean-tokens.json',
            'LaBSE': 'config/qa/sentence-emb/LaBSE.json',
            'mpnet': 'config/qa/sentence-emb/paraphrase-mpnet-base-v2.json',
            'mpnet-multi': 'config/qa/sentence-emb/paraphrase-multilingual-mpnet-base-v2.json'
        }

        # mpnet sota perfromance https://www.sbert.net/docs/pretrained_models.html
        if self.__configs[config] is None:
            config = 'all-mpnet'

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
    By default it loads the mpnet-v2 sentence embedding
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
    query is the query
    k is the top-k results
    :return :   Returns a list with the top-k results. Each item in the list is a tuple that contains the answer
                and its probability
    """

    def ask(self, query, k):
        # Check that model is loaded
        if self.__loaded is False:
            return 'ERROR! FAQ model not loaded!', 1.0

        # Get all available classes
        classes = list(dict(self.__model['classes_vocab']).keys())

        # Ask the model the user query. Get the prediction label and the probabilities
        result = self.__model.compute([query], targets=['y_pred_labels', 'y_pred_probas'])

        # Get the array with all probabilities
        probas = result[1][0]
        # sorted probas
        sorted_probas = sorted(probas, reverse=True)
        # indexes of sorted probas
        sorted_probas_index = sorted(range(len(probas)), key=lambda x: probas[x], reverse=True)

        # Get the answer and its probability
        answer = result[0][0]
        prob = np.amax(result[1])

        # For logging
        print('Query: ' + query)

        results = []
        for i in range(k):
            answer = classes[sorted_probas_index[i]]
            prob = sorted_probas[i]
            item = answer, str(prob)
            results.append(item)
            # Print the result
            print('FAQ[' + str(i) + ']: Answer: ' + answer + ' probability:' + str(prob))

        return 'faq', results
