# FOUNDATION OF RESEARCH AND TECHNOLOGY - HELLAS (FORTH-ICS)
#
# INFORMATION SYSTEMS LABORATORY (ISL)
#
# http://www.ics.forth.gr/isl
#
# LICENCE: TO BE ADDED
#
# Copyright 2021

# The AI4EU chatbot - Class making requests to the Search-API developed by Thales

# author: Papadakos Panagiotis
# e-mail: papadako@ics.forth.gr

# importing the requests library
import requests

#from search_api_response import SearchAPIResults as S_API_R


class SearchAPI:

    """"
    Constructor for search API requests
    Using JSON as a serialization format
    """
    def __init__(self):
        self._SEARCH_API_ENDPOINT = 'https://ws66.tl.teralab-datascience.fr:8000/api/searching/querying'
        self.json = None
        self.headers = {'Content-type': 'application/json'}

    """
    Method that returns the results of a simple web query
    Takes the query and the number of results
    Returns the response as json
    """
    def web_query(self, query, results=1):
        print('Search-API make web search query for ', query)

        # populate the json for the search API POST request
        self.json = SearchAPI.__populate_web_query_data(query, results)

        # Make the POST request
        resp = requests.post(url=self._SEARCH_API_ENDPOINT, json=self.json, headers=self.headers, verify=False)
        return resp.json()

    """
    Method that returns the results of an ai_catalogue query
    PP: TODO Check if we need to support other kind of slots
    """
    def ai_catalogue_query(self, query, results=1,
                           research_area=None,
                           asset_type=None,
                           technical_categories=None,
                           business_categories=None):

        print('Search-API make asset search query for ', query,
              'Research Area: ',  research_area,
                'Asset Type: ', asset_type,
                'Technical Categories: ', technical_categories,
              'Bussiness Categories: ', business_categories)

        # populate the json for the search API POST request
        # Here we are also adding any values for the slots we are tracking
        self.json = SearchAPI.__populate_ai_catalogue_query_data(query, results,
                                                                 research_area=research_area,
                                                                 asset_type=asset_type,
                                                                 technical_categories=technical_categories,
                                                                 business_categories=business_categories)
        # Make the POST request
        resp = requests.post(url=self._SEARCH_API_ENDPOINT, json=self.json, headers=self.headers, verify=False)
        return resp.json()

    @staticmethod
    def __populate_web_query_data(query, results=1):
        # json to be sent to api
        json = {
            'from': 0,
            'size': results,
            'content': query,
            'options': {
                'exclude': ['summary', 'sentiment', 'categories', 'text_suggester', 'lemma_title', 'lemma_content'],
                'disable': ['qa', 'aggregator']
            }
        }

        return json

    """
    method that populates the json for the ai catalogue query
    We need the query, the number of results and any value for the facets of researchArea, assetType, 
    technicalCategories or businessCategories
    """
    @staticmethod
    def __populate_ai_catalogue_query_data(query, results=1,
                                           research_area=None,
                                           asset_type=None,
                                           technical_categories=None,
                                           business_categories=None):
        # Return only results is the ai-catalog
        must = [{'wildcard': {'source_doc_id': 'https://www.ai4europe.eu/research/ai-catalog/*'}}]
        # Check what other filters we want to add
        # researchArea
        if research_area is not None:
            must.append({'match': {'content': research_area}})
        # assetType
        if asset_type is not None:
            must.append({'match': {'content': asset_type}})
        # technicalCategories
        if technical_categories is not None:
            must.append({'match': {'content': technical_categories}})
        # businessCategories
        if business_categories is not None:
            must.append({'match': {'content': business_categories}})

        # json to be sent to api
        json = {
            'from': 0,
            'size': 3,
            'content': query,
            'options': {
                'exclude': ['summary', 'sentiment', 'categories', 'text_suggester', 'lemma_title', 'lemma_content'],
                'disable': ['qa', 'aggregator']
            },
            'add-clause': {
                'type': 'must',
                'clause': {
                    'bool': {
                        'must': must
                    }
                }
            }
        }

        return json

'''
# Just trying out things
search_api = SearchAPI()
response = search_api.web_query('What is Yolo?', 3)
item = S_API_R.get_item(response, 0)
print(S_API_R.get_title(item))
print(S_API_R.get_indexed_document(item))
item = S_API_R.get_item(response, 2)
print(S_API_R.get_title(item))
print(S_API_R.get_indexed_document(item))
#search_api.ai_catalogue_query('Earth Observation dataset?', asset_type='dataset',
#                              business_categories='agriculture',
#                              technical_categories='knowledge representation')

'''