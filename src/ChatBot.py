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


class ChatBot:
    def __init__(self):
        self.__faq = FAQ.FAQ()
        self.__THRESHOLD = 0.01

    # Ask the query to our models
    def ask(self, query):
        # First ask the FAQ model
        label, prob = self.__faq.ask(query)

        # Add a threshold for our probabilities
        # If less than this then do something else
        if prob > self.__THRESHOLD:
            return label, prob
        else:
            return 'I did not understand! Can you please try again?', 1.0

# Now test our chatbot
chatbot = ChatBot()

q = chatbot.ask('any open calls?')
print(q)
