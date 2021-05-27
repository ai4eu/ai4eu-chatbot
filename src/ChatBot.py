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

# author: Papadakos Panagiotis
# e-mail: papadako@ics.forth.gr

import FAQ
import KBQA


class ChatBot:
    def __init__(self):
        self.__faq = FAQ.FAQ()
        self.__kbqa = KBQA.KBQA()
        # I have to finetune this threshold
        self.__THRESHOLD = 0.1

    # Ask the query to our models
    def ask(self, query):
        # First ask the FAQ model
        ans, score = self.__faq.ask(query)

        # Add a threshold for our probabilities
        # If less than this then do something else
        if score > self.__THRESHOLD:
            return ans, score

        # Else we were not able to answer the question using the FAQ module
        # Now use the Knowledge Base Question Answering model
        ans = self.__kbqa.ask(query)

        # Currently KBQA just returns the top-ranked result
        # If empty then return rephrase
        if ans == 'Not Found':
            return ans, 1.0
        else:
            return 'I did not understand! Can you please rephrase?', 1.0

# Now test our chatbot
chatbot = ChatBot()

q = chatbot.ask('Which is the capital of Greece?')
print(q)
