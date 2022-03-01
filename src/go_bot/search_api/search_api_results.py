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
from .dto.search_item_in_focus import SearchItemInFocus


class SearchAPIResults:

    """
    Method that returns the results of a query
    """
    @staticmethod
    def get_items(response) -> [SearchItemInFocus]:
        # Check that we have a correct response
        if response is None \
                or response['results'] is None \
                or response['results']['items'] is None:
            return None

        results = response['results']['items']

        sapi_results = []
        for result in results:
            sapi_result = SearchItemInFocus.with_validation(result)
            if sapi_result is not None:
                sapi_results.append(sapi_result)

        # print(sapi_results)
        return sapi_results

    """
    Method that gets a specific item from result
    """
    @staticmethod
    def get_item_from_results(response, index=1) -> SearchItemInFocus:

        items = SearchAPIResults.get_items(response)

        # Something is wrong with the search api response
        if items is None or len(items) == 0:
            return None

        if len(items) <= index:
            print('Asking index out of array ', index)
            return None

        return items[index]

    """
    Method that gets a specific item from items
    We use indexing from 1 - helps in the context vector computation in state
    """
    @staticmethod
    def get_item_from_items(items, index=1) -> SearchItemInFocus:

        # Something is wrong with the search api response
        if items is None or len(items) == 0:
            return None

        if len(items) <= index:
            print('Asking index out of array ', index)
            return None

        return items[index-1]
