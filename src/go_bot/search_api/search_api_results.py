# FOUNDATION OF RESEARCH AND TECHNOLOGY - HELLAS (FORTH-ICS)
#
# INFORMATION SYSTEMS LABORATORY (ISL)
#
# http://www.ics.forth.gr/isl
#
# LICENCE: TO BE ADDED
#
# Copyright 2021

# The AI4EU chatbot - Class for handling the search API responses

# author: Papadakos Panagiotis
# e-mail: papadako@ics.forth.gr

class SearchAPIResults:

    """
    Method that returns the results of a query
    """
    @staticmethod
    def get_items(response):
        # Check that we have a correct response
        if response is None \
                or response['results'] is None \
                or response['results']['items'] is None:
            return None

        return response['results']['items']

    """
    Method that gets a specific item from result
    """
    @staticmethod
    def get_item_from_results(response, index=0):

        items = SearchAPIResults.get_items(response)

        # Something is wrong with the search api response
        if items is None:
            return None

        if len(items) <= index:
            print('Asking index out of array')
            return None

        return items[index]

    """
    Method that gets a specific item from items
    We use indexing from 1 - helps in the context vector computation in state
    """
    @staticmethod
    def get_item_from_items(items, index=1):

        # Something is wrong with the search api response
        if items is None:
            return None

        if len(items) <= index:
            print('Asking index out of array')
            return None

        return items[index-1]

    """
    Method that returns the source of an item in the response
    """
    @staticmethod
    def get_indexed_document(item):

        if item is None:
            return None
        elif item['_source'] is not None:
            return item['_source']['indexed_document']
        else:
            return None

    """
    Method that returns the title of an item in the response
    """
    @staticmethod
    def get_title(item):

        if item is None:
            return None
        elif item['_source'] is not None and item['_source']['title']:
            return item['_source']['title'][0]
        else:
            return None
