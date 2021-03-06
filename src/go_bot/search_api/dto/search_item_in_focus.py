# SearchItemInFocus is a dictionary
# Get only what we need from the search-API response
# Using new Search API json format
class SearchItemInFocus:
    def __init__(self, response: dict):
        source = response.get('_source')

        if source is None:
            return None

        self.score = response.get('_score')
        self.index = response.get('_index')
        self.id = response.get('_id')
        self.type = response.get('_type')

        if not source.get('source_doc_id'):
            self.url = 'Unknown'
        else:
            self.url = source.get('source_doc_id')

        if not source.get('content'):
            self.content = 'Unknown'
        else:
            self.content = source.get('content')[0]

        # Get summary - is an array of summaries
        if not source.get('summary'):
            self.summary = 'Unfortunately there is no summary for this resource'
        else:
            # get the first one
            self.summary = source.get('summary')[0].get('summary')

        # Get title - is an array of title
        titles = source.get('title')
        if titles is None or len(titles) is 0:
            self.title = 'Untitled'
        else:
            # get the first one
            self.title = source.get('title')[0]

        # Get keywords from categories
        if source.get('categories'):
            self.keywords = self._get_top_keywords(source.get('categories'))
        else:
            self.keywords = []

    @classmethod
    def with_validation(cls, response: dict):
        # Check that there is a source
        source = response.get('_source')

        if source is None:
            return None

        search_item = cls(response)

        return search_item

    """

    """
    @classmethod
    def _get_top_keywords(cls, categories):
        if categories is None:
            return []
        # Holds the keywords and their aggregated scores
        keywords = {}

        # We are interested in normalized document and document keywords and title keywords
        # filtered_categories is an array of dicts with labels and scores
        # parse them all and get the top-3
        filtered_categories = []

        if categories.get('normalized-document'):
            filtered_categories += categories.get('normalized-document')
        if categories.get('document'):
            filtered_categories += categories.get('document')
        if categories.get('title_sentences') and categories.get('title_sentences')[0] \
                and categories.get('title_sentences')[0].get('classes'):
            filtered_categories += categories.get('title_sentences')[0].get('classes')

        # For all sets of keywords and scores
        for keyword_item in filtered_categories:
            # For all keywords
            keyword = keyword_item.get('label')
            if keyword in keywords.keys():
                score = keywords.get(keyword)
                # Add the score of index i
                score += keyword_item.get('score')
                keywords[keyword] = score
            else:
                # Just add the value
                score = keyword_item.get('score')
                keywords[keyword] = score

        # Now sort all keywords based on the scores and return a list of sorted keywords
        print(sorted(keywords, key=keywords.get, reverse=True))
        return sorted(keywords, key=keywords.get, reverse=True)

    def get_score(self):
        return self.score

    def get_index(self):
        return self.index

    def get_id(self):
        return self.id

    def get_url(self):
        return self.url

    def get_content(self):
        return self.content

    def get_title(self):
        return self.title

    def get_summary(self):
        return self.summary

    def get_keywords(self):
        return self.keywords

    def print(self):
        score = 'score: ' + str(self.score) + '\n'
        index = 'index: ' + self.index + '\n'
        id = 'id: ' + self.id + '\n'
        url = 'url: ' + self.url + '\n'
        content = 'content: ' + self.content + '\n'
        title = 'title: ' + self.title + '\n'
        summary = 'summary: ' + self.summary + '\n'
        keywords = 'keywords: ' + str(self.keywords) + '\n'

        print(score, index, id, url, content, title, summary, keywords)

'''
test = {
    "results": {
        "items": [
            {
                "_source": {
                    "summary": [
                        {
                            "summary": "AI4Agri Knowledge Graph is a dataset from Earth Observation Machine Learning models and vineyard data in the form of",
                            "block": "AI4Agri Knowledge Graph | AI4EU  Skip to main content Main navigation AI Community * Organizations * Projects Business & Industry * Industrial Verticals Open Calls Case Studies Research * AI Catalog * Research Bundles Education * Education Catalog Education Initiatives * Education related news Ethics Working Groups Workshop OSAI Services * News & Events News Events Past Events Web Caf\u00e9s _ _ Search User account menu _ _ Log in Menu Breadcrumb 1  Home _ _ 2  Research _ _ 3  AI Assets _ _ 4  AI4Agri Knowledge Graph AI4Agri Knowledge Graph A dataset from Earth Observation Machine Learning models and vineyard data in the form of a Knowledge Graph  Dataset _ _ AI4Agri-KG.zip Developed by Artificial Intelligence Team National and Kapodistrian University of Athens License Mozilla Public License 2.0 MPL-2.0 Main Characteristic This data set has been produced for the AI4Agricultural pilot in the AI4EU project and contains all the data used and produced by that pilot  This pilot is targeting a specific application in the scope of precision agriculture to help predicting the yield and assessing the quality of the production in vineyards using remote sensing data and AI models  We have designed an OWL ontology to model the resources and then interlinked the different data both in the temporal and spatial dimensions  The dataset contains in the form of a Knowledge Graph information coming from Earth Observation Machine Learning models and vineyard data that were used in the AI4Agriculture pilot  The dataset is distributed in RDF N-Triples format  We also provide the OWL ontology that was used to model the data  Technical Categories Knowledge Representation Business Categories Agriculture Keywords Semantic web ArtificialIntelligence Agriculture Smart Farming Earth Observation Vineyards Last updated 24.11.2021 19:12 Detailed Description The Knowledge Graph KG contains information from various sources regarding different attributes of grapes in specific parcels of the sample vineyards  Based on the geospatial information of each sample area we were able to link information coming from Earth Observation Machine Learning models and vineyard data"
                        },
                        {
                            "summary": "The KG contains the following data for the years 2020 and 2021 that were used in the pilot",
                            "block": "The KG contains the following data for the years 2020 and 2021 that were used in the pilot Vineyards data parcels sample areas petiol analysis production soil metadata grape maturity NDVI corrected NDVI values based on Satellite images Drone Images geolocated plant images from the sampling areas Number of Clusters clusters of grapes detected in the plants of the sample areas based on a counting model and drone images Yield and Quality data for each sampling area calculated by a model that uses the above sources along with meteorological data The classes and properties that were created to model and link the data are available in the OWL ontology that is provided along with the KG RDF N-Triples format  Trustworthy AI The AI4Agri Knowledge Graph does not violate the guidelines for Trustworthy AI  GDPR Requirements The AI4Agri Knowledge Graph itself is GDPR compliant  Image Footer About AI4EU Legal notice ai4eu in facebook ai4eu on linkedin ai4eu on twitter ai4eu on youtube EU Footer Image European flag This project has received funding from the European Union 's Horizon 2020 research and innovation program under grant agreement 825619 AI Community * Organizations * Projects Business & Industry * Industrial Verticals Open Calls Case Studies Research * AI Catalog * Research Bundles Education * Education Catalog Education Initiatives * Education related news Ethics Working Groups Workshop OSAI Services * News & Events News Events Past Events Web Caf\u00e9s * Search"
                        }
                    ],
                    "categories": {
                        "normalized-document": [
                            {
                                "score": 0.3604612044424933,
                                "label": "machine learning"
                            },
                            {
                                "score": 0.2477606097028105,
                                "label": "natural language processing"
                            },
                            {
                                "score": 0.14639485070748814,
                                "label": "deep learning"
                            },
                            {
                                "score": 0.09388710876349418,
                                "label": "reinforcement learning"
                            },
                            {
                                "score": 0.05051293851353077,
                                "label": "internet of things"
                            },
                            {
                                "score": 0.03749910794896734,
                                "label": "robotic"
                            },
                            {
                                "score": 0.0318205847797109,
                                "label": "neuromorphic computing"
                            },
                            {
                                "score": 0.03166359514150495,
                                "label": "computer vision"
                            }
                        ],
                        "document": [
                            {
                                "score": 0.35980772886151674,
                                "label": "machine learning"
                            },
                            {
                                "score": 0.24962813071637036,
                                "label": "natural language processing"
                            },
                            {
                                "score": 0.14577034518685747,
                                "label": "deep learning"
                            },
                            {
                                "score": 0.09224448481600665,
                                "label": "reinforcement learning"
                            },
                            {
                                "score": 0.051197681023162675,
                                "label": "internet of things"
                            },
                            {
                                "score": 0.03767691745760796,
                                "label": "robotic"
                            },
                            {
                                "score": 0.032056721959494716,
                                "label": "computer vision"
                            },
                            {
                                "score": 0.03161798997898355,
                                "label": "neuromorphic computing"
                            }
                        ],
                        "content_sentences": [
                            {
                                "sequence": " Skip to main content Main navigation AI Community * Organizations * Projects Business & Industry * Industrial Verticals Open Calls Case Studies Research * AI Catalog * Research Bundles Education * Education Catalog Education Initiatives * Education related news Ethics Working Groups Workshop OSAI Services * News & Events News Events Past Events Web Caf\u00e9s _ _ Search User account menu _ _ Log in Menu Breadcrumb 1",
                                "classes": [
                                    {
                                        "score": 0.3760317881991668,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.2715168460952081,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.22922074862205416,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.04878324308650829,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.027157033806043066,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.018996750760429348,
                                        "label": "robotic"
                                    },
                                    {
                                        "score": 0.014846762977272172,
                                        "label": "computer vision"
                                    },
                                    {
                                        "score": 0.013446826453318048,
                                        "label": "neuromorphic computing"
                                    }
                                ]
                            },
                            {
                                "sequence": " Home _ _ 2",
                                "classes": [
                                    {
                                        "score": 0.4328820940489909,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.15002706167653532,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.10596431948332702,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.09443219506352381,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.07711825736255423,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.06483608153496397,
                                        "label": "robotic"
                                    },
                                    {
                                        "score": 0.03790279531763805,
                                        "label": "computer vision"
                                    },
                                    {
                                        "score": 0.03683719551246668,
                                        "label": "neuromorphic computing"
                                    }
                                ]
                            },
                            {
                                "sequence": " Research _ _ 3",
                                "classes": [
                                    {
                                        "score": 0.25638412066468985,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.25620185107850413,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.20903774109350426,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.08285834657230845,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.06655615096686336,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.04781369950606693,
                                        "label": "robotic"
                                    },
                                    {
                                        "score": 0.043902238662755604,
                                        "label": "neuromorphic computing"
                                    },
                                    {
                                        "score": 0.037245851455307434,
                                        "label": "computer vision"
                                    }
                                ]
                            },
                            {
                                "sequence": " AI Assets _ _ 4",
                                "classes": [
                                    {
                                        "score": 0.3324326685763984,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.24417572649066327,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.10801987939509822,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.1044607812019998,
                                        "label": "robotic"
                                    },
                                    {
                                        "score": 0.07423259665546757,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.05414171615454905,
                                        "label": "neuromorphic computing"
                                    },
                                    {
                                        "score": 0.05074549726669506,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.03179113425912864,
                                        "label": "computer vision"
                                    }
                                ]
                            },
                            {
                                "sequence": " AI4Agri Knowledge Graph AI4Agri Knowledge Graph A dataset from Earth Observation Machine Learning models and vineyard data in the form of a Knowledge Graph",
                                "classes": [
                                    {
                                        "score": 0.8167611697883166,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.06051690350157952,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.04204841799052928,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.04124952289708699,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.015575196728992179,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.013661752293063831,
                                        "label": "neuromorphic computing"
                                    },
                                    {
                                        "score": 0.007512369523955624,
                                        "label": "computer vision"
                                    },
                                    {
                                        "score": 0.0026746672764759657,
                                        "label": "robotic"
                                    }
                                ]
                            },
                            {
                                "sequence": " Dataset _ _ AI4Agri-KG.zip Developed by Artificial Intelligence Team National and Kapodistrian University of Athens License Mozilla Public License 2.0 MPL-2.0 Main Characteristic This data set has been produced for the AI4Agricultural pilot in the AI4EU project and contains all the data used and produced by that pilot",
                                "classes": [
                                    {
                                        "score": 0.28062452639806845,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.20819421398684299,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.15845685943606114,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.11369438397128699,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.10039141998273395,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.06048021078101799,
                                        "label": "neuromorphic computing"
                                    },
                                    {
                                        "score": 0.0476156348059339,
                                        "label": "robotic"
                                    },
                                    {
                                        "score": 0.0305427506380546,
                                        "label": "computer vision"
                                    }
                                ]
                            },
                            {
                                "sequence": " This pilot is targeting a specific application in the scope of precision agriculture to help predicting the yield and assessing the quality of the production in vineyards using remote sensing data and AI models",
                                "classes": [
                                    {
                                        "score": 0.6669129243244466,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.1173325218994049,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.06414931536321555,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.050138159262737084,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.039827339937884544,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.026309873346113125,
                                        "label": "computer vision"
                                    },
                                    {
                                        "score": 0.02588031804555595,
                                        "label": "neuromorphic computing"
                                    },
                                    {
                                        "score": 0.009449547820642201,
                                        "label": "robotic"
                                    }
                                ]
                            },
                            {
                                "sequence": " We have designed an OWL ontology to model the resources and then interlinked the different data both in the temporal and spatial dimensions",
                                "classes": [
                                    {
                                        "score": 0.2512439789715233,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.23282790655846508,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.17223267161055678,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.09550783858670281,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.07331258317587225,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.07030926349574446,
                                        "label": "robotic"
                                    },
                                    {
                                        "score": 0.055424978861298546,
                                        "label": "neuromorphic computing"
                                    },
                                    {
                                        "score": 0.04914077873983682,
                                        "label": "computer vision"
                                    }
                                ]
                            },
                            {
                                "sequence": " The dataset contains in the form of a Knowledge Graph information coming from Earth Observation Machine Learning models and vineyard data that were used in the AI4Agriculture pilot",
                                "classes": [
                                    {
                                        "score": 0.7822103351207023,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.07897164886035767,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.04859848792148635,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.04617141137045634,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.015437023423810311,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.01356468347306548,
                                        "label": "computer vision"
                                    },
                                    {
                                        "score": 0.010905983886574303,
                                        "label": "neuromorphic computing"
                                    },
                                    {
                                        "score": 0.004140425943547282,
                                        "label": "robotic"
                                    }
                                ]
                            },
                            {
                                "sequence": " The dataset is distributed in RDF N-Triples format",
                                "classes": [
                                    {
                                        "score": 0.2487986037147251,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.24444103273330028,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.24281418798005078,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.08329030922200309,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.06669626895775341,
                                        "label": "robotic"
                                    },
                                    {
                                        "score": 0.05402715620564815,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.03235259816159259,
                                        "label": "computer vision"
                                    },
                                    {
                                        "score": 0.027579843024926595,
                                        "label": "neuromorphic computing"
                                    }
                                ]
                            },
                            {
                                "sequence": " We also provide the OWL ontology that was used to model the data",
                                "classes": [
                                    {
                                        "score": 0.3677269203123184,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.18763012063358558,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.14583054941763512,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.11196270096646643,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.06015955055761334,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.05860951943845564,
                                        "label": "robotic"
                                    },
                                    {
                                        "score": 0.03872469641394698,
                                        "label": "computer vision"
                                    },
                                    {
                                        "score": 0.02935594225997853,
                                        "label": "neuromorphic computing"
                                    }
                                ]
                            },
                            {
                                "sequence": " Technical Categories Knowledge Representation Business Categories Agriculture Keywords Semantic web ArtificialIntelligence Agriculture Smart Farming Earth Observation Vineyards Last updated 24.11.2021 19:12 Detailed Description The Knowledge Graph KG contains information from various sources regarding different attributes of grapes in specific parcels of the sample vineyards",
                                "classes": [
                                    {
                                        "score": 0.4421876605321573,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.1640127387868247,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.15444335122746436,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.09913163749664008,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.07193015130562552,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.026979166622524402,
                                        "label": "neuromorphic computing"
                                    },
                                    {
                                        "score": 0.02493949500359001,
                                        "label": "computer vision"
                                    },
                                    {
                                        "score": 0.01637579902517364,
                                        "label": "robotic"
                                    }
                                ]
                            },
                            {
                                "sequence": " Based on the geospatial information of each sample area we were able to link information coming from Earth Observation Machine Learning models and vineyard data",
                                "classes": [
                                    {
                                        "score": 0.7502579630410797,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.06942476293234676,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.06107845996960531,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.0595699446826592,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.026462080893357675,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.014244833691703257,
                                        "label": "computer vision"
                                    },
                                    {
                                        "score": 0.009913273337771415,
                                        "label": "robotic"
                                    },
                                    {
                                        "score": 0.009048681451476664,
                                        "label": "neuromorphic computing"
                                    }
                                ]
                            },
                            {
                                "sequence": " The KG contains the following data for the years 2020 and 2021 that were used in the pilot Vineyards data parcels sample areas petiol analysis production soil metadata grape maturity NDVI corrected NDVI values based on Satellite images Drone Images geolocated plant images from the sampling areas Number of Clusters clusters of grapes detected in the plants of the sample areas based on a counting model and drone images Yield and Quality data for each sampling area calculated by a model that uses the above sources along with meteorological data The classes and properties that were created to model and link the data are available in the OWL ontology that is provided along with the KG RDF N-Triples format",
                                "classes": [
                                    {
                                        "score": 0.4971477297861287,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.10576436594275644,
                                        "label": "computer vision"
                                    },
                                    {
                                        "score": 0.09811730235606443,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.09347156506717555,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.07682859753307743,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.07523587309807565,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.0296221457216178,
                                        "label": "neuromorphic computing"
                                    },
                                    {
                                        "score": 0.023812420495103962,
                                        "label": "robotic"
                                    }
                                ]
                            },
                            {
                                "sequence": " Trustworthy AI The AI4Agri Knowledge Graph does not violate the guidelines for Trustworthy AI",
                                "classes": [
                                    {
                                        "score": 0.38221030472526446,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.2130028226751971,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.14721592234152628,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.11012120174790233,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.04717246378631389,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.03787223475854661,
                                        "label": "robotic"
                                    },
                                    {
                                        "score": 0.033057541647095116,
                                        "label": "neuromorphic computing"
                                    },
                                    {
                                        "score": 0.02934750831815419,
                                        "label": "computer vision"
                                    }
                                ]
                            },
                            {
                                "sequence": " GDPR Requirements The AI4Agri Knowledge Graph itself is GDPR compliant",
                                "classes": [
                                    {
                                        "score": 0.3997630247091425,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.16179414936150924,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.10947841252828214,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.10838014968693725,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.08056026165607036,
                                        "label": "robotic"
                                    },
                                    {
                                        "score": 0.06272346231528601,
                                        "label": "neuromorphic computing"
                                    },
                                    {
                                        "score": 0.0427190257461333,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.03458151399663924,
                                        "label": "computer vision"
                                    }
                                ]
                            },
                            {
                                "sequence": " Image Footer About AI4EU Legal notice ai4eu in facebook ai4eu on linkedin ai4eu on twitter ai4eu on youtube EU Footer Image European flag This project has received funding from the European Union 's Horizon 2020 research and innovation program under grant agreement 825619",
                                "classes": [
                                    {
                                        "score": 0.450323582084582,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.1369611099818992,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.12178010433809842,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.10708198634347786,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.07741953414673884,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.04307201349770259,
                                        "label": "computer vision"
                                    },
                                    {
                                        "score": 0.03652615849637857,
                                        "label": "robotic"
                                    },
                                    {
                                        "score": 0.026835511111122548,
                                        "label": "neuromorphic computing"
                                    }
                                ]
                            },
                            {
                                "sequence": "AI Community * Organizations * Projects Business & Industry * Industrial Verticals Open Calls Case Studies Research * AI Catalog * Research Bundles Education * Education Catalog Education Initiatives * Education related news Ethics Working Groups Workshop OSAI Services * News & Events News Events Past Events Web Caf\u00e9s * Search",
                                "classes": [
                                    {
                                        "score": 0.9019573898466736,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.028608077548094836,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.025387374960150685,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.022282201544879476,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.008680708207165124,
                                        "label": "internet of things"
                                    },
                                    {
                                        "score": 0.0051474824603195395,
                                        "label": "computer vision"
                                    },
                                    {
                                        "score": 0.004266008861090585,
                                        "label": "neuromorphic computing"
                                    },
                                    {
                                        "score": 0.0036707565716261,
                                        "label": "robotic"
                                    }
                                ]
                            }
                        ],
                        "title_sentences": [
                            {
                                "sequence": "AI4Agri Knowledge Graph | AI4EU",
                                "classes": [
                                    {
                                        "score": 0.5003741194572322,
                                        "label": "machine learning"
                                    },
                                    {
                                        "score": 0.22943117449509753,
                                        "label": "deep learning"
                                    },
                                    {
                                        "score": 0.09635753321842493,
                                        "label": "natural language processing"
                                    },
                                    {
                                        "score": 0.08238055338446798,
                                        "label": "reinforcement learning"
                                    },
                                    {
                                        "score": 0.036592285934969236,
                                        "label": "neuromorphic computing"
                                    },
                                    {
                                        "score": 0.03204621001362294,
                                        "label": "computer vision"
                                    },
                                    {
                                        "score": 0.011527886611867837,
                                        "label": "robotic"
                                    },
                                    {
                                        "score": 0.011290236884317384,
                                        "label": "internet of things"
                                    }
                                ]
                            }
                        ]
                    },
                    "indexed_document": "https:\/\/www.ai4europe.eu\/research\/ai-catalog\/ai4agri-knowledge-graph",
                    "title": [
                        "AI4Agri Knowledge Graph | AI4EU"
                    ],
                    "kg": [
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "Education",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 28,
                                "content": "Education"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "relate",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 51,
                                "content": "related"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "news",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 36,
                                "content": "news"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "dataset _ _ ai4agri-kg.zip develop by Artificial Intelligence Team - National and Kapodistrian University of Athens License Mozilla Public License 2.0 ( mpl-2.0 ) main Characteristic this data set",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_subject",
                                "class": 34,
                                "content": "Dataset _ _ AI4Agri-KG.zip Developed by Artificial Intelligence Team - National and Kapodistrian University of Athens License Mozilla Public License 2.0 ( MPL-2.0 ) Main Characteristic This data set"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "have be produce",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_verb",
                                "class": 11,
                                "content": "has been produced"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "datum",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_object",
                                "class": 37,
                                "content": "data"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "dataset _ _ ai4agri-kg.zip develop by Artificial Intelligence Team - National and Kapodistrian University of Athens License Mozilla Public License 2.0 ( mpl-2.0 ) main Characteristic this data set",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_subject",
                                "class": 34,
                                "content": "Dataset _ _ AI4Agri-KG.zip Developed by Artificial Intelligence Team - National and Kapodistrian University of Athens License Mozilla Public License 2.0 ( MPL-2.0 ) Main Characteristic This data set"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "contain",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_verb",
                                "class": 21,
                                "content": "contains"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "datum",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_object",
                                "class": 37,
                                "content": "data"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai4agri-kg.zip",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 6,
                                "content": "AI4Agri-KG.zip"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "develop by",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 54,
                                "content": "Developed by"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "Artificial Intelligence Team",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 13,
                                "content": "Artificial Intelligence Team"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "main Characteristic this data set",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 34,
                                "content": "Main Characteristic This data set"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "have be produce for",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 11,
                                "content": "has been produced for"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "AI4Agricultural pilot in the AI4EU project",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 34,
                                "content": "AI4Agricultural pilot in the AI4EU project"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "datum",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 53,
                                "content": "data"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "use and produce by",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 9,
                                "content": "used and produced by"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "pilot",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 39,
                                "content": "pilot"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "pilot",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 10,
                                "content": "pilot"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "be target",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 18,
                                "content": "is targeting"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "specific application in the scope of precision agriculture",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 3,
                                "content": "specific application in the scope of precision agriculture"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "pilot",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_subject",
                                "class": 10,
                                "content": "pilot"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "be target",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_verb",
                                "class": 18,
                                "content": "is targeting"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "application",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_object",
                                "class": 49,
                                "content": "application"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "quality of the production in vineyard",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 34,
                                "content": "quality of the production in vineyards"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "use",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 5,
                                "content": "using"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "AI",
                                "positions": [
                                    -1
                                ],
                                "label": "organization",
                                "class": 13,
                                "content": "AI"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "OWL ontology",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 34,
                                "content": "OWL ontology"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "model",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 60,
                                "content": "model"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "resource",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 36,
                                "content": "resources"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "OWL",
                                "positions": [
                                    -1
                                ],
                                "label": "organization",
                                "class": 6,
                                "content": "OWL"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "model",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 60,
                                "content": "model"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "resource",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 36,
                                "content": "resources"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "dataset",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 29,
                                "content": "dataset"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "contain in",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 21,
                                "content": "contains in"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "form of a Knowledge Graph information",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 34,
                                "content": "form of a Knowledge Graph information"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "form of a Knowledge Graph information",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 34,
                                "content": "form of a Knowledge Graph information"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "come from",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 42,
                                "content": "coming from"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "Earth Observation",
                                "positions": [
                                    -1
                                ],
                                "label": "location",
                                "class": 34,
                                "content": "Earth Observation"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "dataset",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 29,
                                "content": "dataset"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "be distribute in",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 35,
                                "content": "is distributed in"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "RDF",
                                "positions": [
                                    -1
                                ],
                                "label": "organization",
                                "class": 21,
                                "content": "RDF"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "Observation Vineyards",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_subject",
                                "class": 34,
                                "content": "Observation Vineyards"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "update",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_verb",
                                "class": 30,
                                "content": "updated"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "Description",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_object",
                                "class": 7,
                                "content": "Description"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "Knowledge Graph",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_subject",
                                "class": 6,
                                "content": "Knowledge Graph"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "contain",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_verb",
                                "class": 21,
                                "content": "contains"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "information",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_object",
                                "class": 36,
                                "content": "information"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "information from various source",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 2,
                                "content": "information from various sources"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "regard",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 41,
                                "content": "regarding"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "different attribute of grape in specific parcel of the sample vineyard",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 3,
                                "content": "different attributes of grapes in specific parcels of the sample vineyards"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "able",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 14,
                                "content": "able"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "link",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 51,
                                "content": "link"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "information",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 36,
                                "content": "information"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "information",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 29,
                                "content": "information"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "come from",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 42,
                                "content": "coming from"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "Earth Observation",
                                "positions": [
                                    -1
                                ],
                                "label": "location",
                                "class": 34,
                                "content": "Earth Observation"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "KG",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_subject",
                                "class": 6,
                                "content": "KG"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "contain",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_verb",
                                "class": 21,
                                "content": "contains"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "datum for the year 2020 and 2021 , that be use in the pilot : * vineyard datum ( parcel , sample area , petiol analysis , production , soil metadata , grape maturity ) * NDVI ( correct NDVI value base on Satellite image ) * drone image ( geolocate plant image from the sample area ) * number of cluster ( cluster of grape detect in the plant of the sample area , base on a counting model and drone image ) * yield and Quality ( datum for each sample area calculate by a model that use the above source along with meteorological datum ) the class and property",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_object",
                                "class": 35,
                                "content": "data for the years 2020 and 2021 , that were used in the pilot : * Vineyards data ( parcels , sample areas , petiol analysis , production , soil metadata , grape maturity ) * NDVI ( corrected NDVI values based on Satellite images ) * Drone Images ( geolocated plant images from the sampling areas ) * Number of Clusters ( clusters of grapes detected in the plants of the sample areas , based on a counting model and drone images ) * Yield and Quality ( data for each sampling area calculated by a model that uses the above sources along with meteorological data ) The classes and properties"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "NDVI value",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 6,
                                "content": "NDVI values"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "base on",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 28,
                                "content": "based on"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "Satellite image",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 39,
                                "content": "Satellite images"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "drone image",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_subject",
                                "class": 6,
                                "content": "Drone Images"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "geolocate",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_verb",
                                "class": 60,
                                "content": "geolocated"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "plant image",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_object",
                                "class": 59,
                                "content": "plant images"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "cluster of grape",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 34,
                                "content": "clusters of grapes"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "detect in",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 24,
                                "content": "detected in"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "plant of the sample area",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 38,
                                "content": "plants of the sample areas"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "area",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 37,
                                "content": "area"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "calculate by",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 41,
                                "content": "calculated by"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "model",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 16,
                                "content": "model"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "class and property",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 24,
                                "content": "classes and properties"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "to model",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_infinitive_verb",
                                "class": 60,
                                "content": "to model"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "datum",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 37,
                                "content": "data"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "datum",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 53,
                                "content": "data"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "be",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 47,
                                "content": "are"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "available in the owl ontology",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 34,
                                "content": "available in the OWL ontology"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "AI4Agri Knowledge Graph",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_subject",
                                "class": 40,
                                "content": "AI4Agri Knowledge Graph"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "do not violate",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_verb",
                                "class": 37,
                                "content": "does not violate"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "guideline",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_object",
                                "class": 36,
                                "content": "guidelines"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "GDPR",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 6,
                                "content": "GDPR"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "requirement",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 10,
                                "content": "Requirements"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "AI4Agri Knowledge Graph",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 13,
                                "content": "AI4Agri Knowledge Graph"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai4eu on youtube EU Footer Image european flag this project",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 34,
                                "content": "ai4eu on youtube EU Footer Image European flag This project"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "have receive",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 39,
                                "content": "has received"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "Union",
                                "positions": [
                                    -1
                                ],
                                "label": "location.city",
                                "class": 11,
                                "content": "Union"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "EU",
                                "positions": [
                                    -1
                                ],
                                "label": "organization",
                                "class": 1,
                                "content": "EU"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "have receive",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 39,
                                "content": "has received"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "Union",
                                "positions": [
                                    -1
                                ],
                                "label": "location.city",
                                "class": 11,
                                "content": "Union"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai4eu on youtube EU Footer Image european flag this project",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_subject",
                                "class": 34,
                                "content": "ai4eu on youtube EU Footer Image European flag This project"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "have receive",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_verb",
                                "class": 39,
                                "content": "has received"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "funding",
                                "positions": [
                                    -1
                                ],
                                "label": "dep_object",
                                "class": 36,
                                "content": "funding"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "Education",
                                "positions": [
                                    -1
                                ],
                                "label": "industry",
                                "class": 28,
                                "content": "Education"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "relate",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_verb_phrase",
                                "class": 51,
                                "content": "related"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "news",
                                "positions": [
                                    -1
                                ],
                                "label": "pattern_syntagm_or_prep_group",
                                "class": 36,
                                "content": "news"
                            },
                            "field_type": "content"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "resource",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 26,
                                "content": "resource"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "number",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 62,
                                "content": "number"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai community",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 49,
                                "content": "ai community"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "national",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 13,
                                "content": "national"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "web caf\u00e9s",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 6,
                                "content": "web caf\u00e9s"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "industrial verticals",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "industrial verticals"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "data set have be produce",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 19,
                                "content": "data set have be produce"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "athens license mozilla public license 2.0",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 9,
                                "content": "athens license mozilla public license 2.0"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "precision agriculture to help predict",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 40,
                                "content": "precision agriculture to help predict"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai4agri knowledge graph ai4agri knowledge graph a dataset",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 9,
                                "content": "ai4agri knowledge graph ai4agri knowledge graph a dataset"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "dataset be distribute",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 19,
                                "content": "dataset be distribute"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "case studies",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 35,
                                "content": "case studies"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "_ _ search user account menu",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 43,
                                "content": "_ _ search user account menu"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "open call",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 14,
                                "content": "open call"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "news",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 32,
                                "content": "news"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "datum use",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 59,
                                "content": "datum use"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "legal notice",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 27,
                                "content": "legal notice"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "datum be available",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 59,
                                "content": "datum be available"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "counting model",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 29,
                                "content": "counting model"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai4agri knowledge graph do not violate",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 9,
                                "content": "ai4agri knowledge graph do not violate"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "various source regard different attribute",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 21,
                                "content": "various source regard different attribute"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "vineyard use remote sensing datum",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 9,
                                "content": "vineyard use remote sensing datum"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "organizations",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 49,
                                "content": "organizations"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai4agri knowledge graph | ai4eu",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 9,
                                "content": "ai4agri knowledge graph | ai4eu"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "education initiatives",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 42,
                                "content": "education initiatives"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "form",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 29,
                                "content": "form"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "main content main navigation",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 9,
                                "content": "main content main navigation"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "pilot be target",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "pilot be target"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "dataset contain",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 19,
                                "content": "dataset contain"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "log",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 25,
                                "content": "log"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "event",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 32,
                                "content": "event"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "search",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 35,
                                "content": "search"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "main characteristic",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 26,
                                "content": "main characteristic"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "sample vineyard",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "sample vineyard"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "research bundle",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 35,
                                "content": "research bundle"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "grape maturity",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "grape maturity"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "contain information",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 19,
                                "content": "contain information"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai4eu project",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 52,
                                "content": "ai4eu project"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "artificial intelligence team",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 40,
                                "content": "artificial intelligence team"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ndvi",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 55,
                                "content": "ndvi"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "innovation program",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 33,
                                "content": "innovation program"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "business",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 49,
                                "content": "business"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "owl ontology to model",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "owl ontology to model"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "guideline",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 27,
                                "content": "guideline"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "be gdpr compliant",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 11,
                                "content": "be gdpr compliant"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai model",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 29,
                                "content": "ai model"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "project have receive funding",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 11,
                                "content": "project have receive funding"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "projects",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 5,
                                "content": "projects"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "drone image",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 6,
                                "content": "drone image"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "pilot",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 10,
                                "content": "pilot"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "soil metadata",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "soil metadata"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "twitter",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 2,
                                "content": "twitter"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "property",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 15,
                                "content": "property"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "quality",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 15,
                                "content": "quality"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "research * ai catalog",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 36,
                                "content": "research * ai catalog"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "trustworthy ai",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 3,
                                "content": "trustworthy ai"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "meteorological datum",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 59,
                                "content": "meteorological datum"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "vineyard datum",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 59,
                                "content": "vineyard datum"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "_",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 30,
                                "content": "_"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "parcel",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 7,
                                "content": "parcel"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "past event",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 17,
                                "content": "past event"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "correct ndvi value base",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "correct ndvi value base"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "youtube eu footer image european flag",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 9,
                                "content": "youtube eu footer image european flag"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "knowledge graph information come",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 40,
                                "content": "knowledge graph information come"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "kg contain",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 55,
                                "content": "kg contain"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai4agricultural pilot",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 9,
                                "content": "ai4agricultural pilot"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "model",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 29,
                                "content": "model"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "working groups",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 49,
                                "content": "working groups"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai4agri knowledge graph",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "ai4agri knowledge graph"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "kapodistrian university",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 4,
                                "content": "kapodistrian university"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai4eu",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 52,
                                "content": "ai4eu"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "gdpr requirement",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "gdpr requirement"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "plant",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 12,
                                "content": "plant"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "owl ontology",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "owl ontology"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "facebook",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 51,
                                "content": "facebook"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "education * education catalog",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 36,
                                "content": "education * education catalog"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "specific parcel",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "specific parcel"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "earth observation",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "earth observation"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai4agriculture pilot",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 9,
                                "content": "ai4agriculture pilot"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "specific application",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 52,
                                "content": "specific application"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "machine learning model",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 16,
                                "content": "machine learning model"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ethic",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 27,
                                "content": "ethic"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "yield",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 15,
                                "content": "yield"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "be able to link information come",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 11,
                                "content": "be able to link information come"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "workshop",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 49,
                                "content": "workshop"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "above source",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 26,
                                "content": "above source"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "satellite image",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "satellite image"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "ai4agri-kg.zip develop",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 9,
                                "content": "ai4agri-kg.zip develop"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "knowledge graph",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 10,
                                "content": "knowledge graph"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "grape detect",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "grape detect"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "different datum",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 59,
                                "content": "different datum"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "sample area calculate",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "sample area calculate"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "events",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 32,
                                "content": "events"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "scope",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 12,
                                "content": "scope"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "services * news",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 36,
                                "content": "services * news"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "geospatial information",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 19,
                                "content": "geospatial information"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "cluster",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 37,
                                "content": "cluster"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "production",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 15,
                                "content": "production"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "sample area",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 29,
                                "content": "sample area"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "spatial dimension",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "spatial dimension"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "class",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 37,
                                "content": "class"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "geolocate plant image",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "geolocate plant image"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "linkedin",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 27,
                                "content": "linkedin"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "kg",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 55,
                                "content": "kg"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "technical categories knowledge representation business categories agriculture keywords semantic web artificialintelligence agriculture smart farming earth observation vineyards",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 40,
                                "content": "technical categories knowledge representation business categories agriculture keywords semantic web artificialintelligence agriculture smart farming earth observation vineyards"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "image footer",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "image footer"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "petiol analysis",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 38,
                                "content": "petiol analysis"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "follow datum",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 59,
                                "content": "follow datum"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "industry",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 49,
                                "content": "industry"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "datum",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 53,
                                "content": "datum"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "grape",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 7,
                                "content": "grape"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "osai",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 41,
                                "content": "osai"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "education relate news",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 27,
                                "content": "education relate news"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "temporal",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 28,
                                "content": "temporal"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "rdf n-triples format",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": 9,
                                "content": "rdf n-triples format"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    97
                                ],
                                "label": "",
                                "class": -1,
                                "content": "AI"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "organization",
                                "positions": [
                                    -1
                                ],
                                "label": "organization",
                                "class": -1,
                                "content": "organization"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    112,
                                    113
                                ],
                                "label": "",
                                "class": -1,
                                "content": "Earth Observation"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "location",
                                "positions": [
                                    -1
                                ],
                                "label": "location",
                                "class": -1,
                                "content": "location"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    210
                                ],
                                "label": "",
                                "class": -1,
                                "content": "AI"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "organization",
                                "positions": [
                                    -1
                                ],
                                "label": "organization",
                                "class": -1,
                                "content": "organization"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    217
                                ],
                                "label": "",
                                "class": -1,
                                "content": "OWL"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "organization",
                                "positions": [
                                    -1
                                ],
                                "label": "organization",
                                "class": -1,
                                "content": "organization"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    250,
                                    251
                                ],
                                "label": "",
                                "class": -1,
                                "content": "Earth Observation"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "location",
                                "positions": [
                                    -1
                                ],
                                "label": "location",
                                "class": -1,
                                "content": "location"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    272
                                ],
                                "label": "",
                                "class": -1,
                                "content": "RDF"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "organization",
                                "positions": [
                                    -1
                                ],
                                "label": "organization",
                                "class": -1,
                                "content": "organization"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    290,
                                    291,
                                    292,
                                    293,
                                    294,
                                    295,
                                    296,
                                    297,
                                    298
                                ],
                                "label": "",
                                "class": -1,
                                "content": "Technical Categories Knowledge Representation Business Categories Agriculture Keywords Semantic"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "organization",
                                "positions": [
                                    -1
                                ],
                                "label": "organization",
                                "class": -1,
                                "content": "organization"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    309,
                                    310,
                                    311,
                                    312,
                                    313,
                                    314,
                                    315,
                                    316
                                ],
                                "label": "",
                                "class": -1,
                                "content": "24.11.2021 - 19:12 Detailed Description The Knowledge Graph"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "date",
                                "positions": [
                                    -1
                                ],
                                "label": "date",
                                "class": -1,
                                "content": "date"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    356,
                                    357
                                ],
                                "label": "",
                                "class": -1,
                                "content": "Earth Observation"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "location",
                                "positions": [
                                    -1
                                ],
                                "label": "location",
                                "class": -1,
                                "content": "location"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    372,
                                    373,
                                    374,
                                    375,
                                    376
                                ],
                                "label": "",
                                "class": -1,
                                "content": "the years 2020 and 2021"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "date",
                                "positions": [
                                    -1
                                ],
                                "label": "date",
                                "class": -1,
                                "content": "date"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    505
                                ],
                                "label": "",
                                "class": -1,
                                "content": "RDF"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "organization",
                                "positions": [
                                    -1
                                ],
                                "label": "organization",
                                "class": -1,
                                "content": "organization"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    522,
                                    523
                                ],
                                "label": "",
                                "class": -1,
                                "content": "Trustworthy AI"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "organization",
                                "positions": [
                                    -1
                                ],
                                "label": "organization",
                                "class": -1,
                                "content": "organization"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    560
                                ],
                                "label": "",
                                "class": -1,
                                "content": "EU"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "organization",
                                "positions": [
                                    -1
                                ],
                                "label": "organization",
                                "class": -1,
                                "content": "organization"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    573
                                ],
                                "label": "",
                                "class": -1,
                                "content": "Union"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "location.city",
                                "positions": [
                                    -1
                                ],
                                "label": "location.city",
                                "class": -1,
                                "content": "location.city"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    615
                                ],
                                "label": "",
                                "class": -1,
                                "content": "Education"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "industry",
                                "positions": [
                                    -1
                                ],
                                "label": "industry",
                                "class": -1,
                                "content": "industry"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    617
                                ],
                                "label": "",
                                "class": -1,
                                "content": "Education"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "industry",
                                "positions": [
                                    -1
                                ],
                                "label": "industry",
                                "class": -1,
                                "content": "industry"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    620
                                ],
                                "label": "",
                                "class": -1,
                                "content": "Education"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "industry",
                                "positions": [
                                    -1
                                ],
                                "label": "industry",
                                "class": -1,
                                "content": "industry"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    623
                                ],
                                "label": "",
                                "class": -1,
                                "content": "Education"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:instanceof",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:instanceof"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "industry",
                                "positions": [
                                    -1
                                ],
                                "label": "industry",
                                "class": -1,
                                "content": "industry"
                            },
                            "field_type": "named-entity"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    0,
                                    1,
                                    2,
                                    3,
                                    4
                                ],
                                "label": "",
                                "class": -1,
                                "content": "ai4agri knowledge graph | ai4eu"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    290,
                                    291,
                                    292,
                                    293,
                                    294,
                                    295,
                                    296,
                                    297,
                                    298,
                                    299,
                                    300,
                                    301,
                                    302,
                                    303,
                                    304,
                                    305,
                                    306
                                ],
                                "label": "",
                                "class": -1,
                                "content": "technical categories knowledge representation business categories agriculture keywords semantic web artificialintelligence agriculture smart farming earth observation vineyards"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    103,
                                    104,
                                    105,
                                    106,
                                    107,
                                    108,
                                    109,
                                    110
                                ],
                                "label": "",
                                "class": -1,
                                "content": "ai4agri knowledge graph ai4agri knowledge graph a dataset"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    513,
                                    514,
                                    515,
                                    516,
                                    517,
                                    518
                                ],
                                "label": "",
                                "class": -1,
                                "content": "ai4agri knowledge graph do not violate"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    189,
                                    190,
                                    191,
                                    192,
                                    193
                                ],
                                "label": "",
                                "class": -1,
                                "content": "precision agriculture to help predict"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    559,
                                    560,
                                    561,
                                    562,
                                    563,
                                    564
                                ],
                                "label": "",
                                "class": -1,
                                "content": "youtube eu footer image european flag"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    72,
                                    73,
                                    74,
                                    75,
                                    76,
                                    77
                                ],
                                "label": "",
                                "class": -1,
                                "content": "_ _ search user account menu"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    349,
                                    350,
                                    351,
                                    352,
                                    353,
                                    354
                                ],
                                "label": "",
                                "class": -1,
                                "content": "be able to link information come"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    152,
                                    153,
                                    154,
                                    155,
                                    156
                                ],
                                "label": "",
                                "class": -1,
                                "content": "data set have be produce"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    323,
                                    324,
                                    325,
                                    326,
                                    327
                                ],
                                "label": "",
                                "class": -1,
                                "content": "various source regard different attribute"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    245,
                                    246,
                                    247,
                                    248
                                ],
                                "label": "",
                                "class": -1,
                                "content": "knowledge graph information come"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    356,
                                    357
                                ],
                                "label": "",
                                "class": -1,
                                "content": "earth observation"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    204,
                                    205,
                                    206,
                                    207,
                                    208
                                ],
                                "label": "",
                                "class": -1,
                                "content": "vineyard use remote sensing datum"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    528,
                                    529,
                                    530
                                ],
                                "label": "",
                                "class": -1,
                                "content": "ai4agri knowledge graph"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    566,
                                    567,
                                    568,
                                    569
                                ],
                                "label": "",
                                "class": -1,
                                "content": "project have receive funding"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    2,
                                    3,
                                    4,
                                    5
                                ],
                                "label": "",
                                "class": -1,
                                "content": "main content main navigation"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    408,
                                    409,
                                    410,
                                    411
                                ],
                                "label": "",
                                "class": -1,
                                "content": "correct ndvi value base"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    615,
                                    616,
                                    617,
                                    618
                                ],
                                "label": "",
                                "class": -1,
                                "content": "education * education catalog"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    217,
                                    218,
                                    219,
                                    220
                                ],
                                "label": "",
                                "class": -1,
                                "content": "owl ontology to model"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    607,
                                    608,
                                    609,
                                    610
                                ],
                                "label": "",
                                "class": -1,
                                "content": "research * ai catalog"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    315,
                                    316
                                ],
                                "label": "",
                                "class": -1,
                                "content": "knowledge graph"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    649,
                                    650
                                ],
                                "label": "",
                                "class": -1,
                                "content": "web caf\u00e9s"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    268,
                                    269,
                                    270
                                ],
                                "label": "",
                                "class": -1,
                                "content": "dataset be distribute"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    532,
                                    533,
                                    534
                                ],
                                "label": "",
                                "class": -1,
                                "content": "be gdpr compliant"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    490,
                                    491,
                                    492
                                ],
                                "label": "",
                                "class": -1,
                                "content": "datum be available"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    594
                                ],
                                "label": "",
                                "class": -1,
                                "content": "business"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    636,
                                    637,
                                    638
                                ],
                                "label": "",
                                "class": -1,
                                "content": "services * news"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    143,
                                    144,
                                    145
                                ],
                                "label": "",
                                "class": -1,
                                "content": "athens license mozilla public license 2.0"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    505,
                                    506,
                                    507
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rdf n-triples format"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    179,
                                    180,
                                    181
                                ],
                                "label": "",
                                "class": -1,
                                "content": "pilot be target"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    623,
                                    624,
                                    625
                                ],
                                "label": "",
                                "class": -1,
                                "content": "education relate news"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    420,
                                    421,
                                    422
                                ],
                                "label": "",
                                "class": -1,
                                "content": "geolocate plant image"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    462,
                                    463,
                                    464
                                ],
                                "label": "",
                                "class": -1,
                                "content": "sample area calculate"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    536,
                                    537
                                ],
                                "label": "",
                                "class": -1,
                                "content": "image footer"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    238,
                                    239
                                ],
                                "label": "",
                                "class": -1,
                                "content": "dataset contain"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    495,
                                    496
                                ],
                                "label": "",
                                "class": -1,
                                "content": "owl ontology"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    170,
                                    171
                                ],
                                "label": "",
                                "class": -1,
                                "content": "datum use"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    227,
                                    228
                                ],
                                "label": "",
                                "class": -1,
                                "content": "different datum"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    320,
                                    321
                                ],
                                "label": "",
                                "class": -1,
                                "content": "contain information"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    341,
                                    342
                                ],
                                "label": "",
                                "class": -1,
                                "content": "geospatial information"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    471,
                                    472
                                ],
                                "label": "",
                                "class": -1,
                                "content": "above source"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    386,
                                    387
                                ],
                                "label": "",
                                "class": -1,
                                "content": "vineyard datum"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    149,
                                    150
                                ],
                                "label": "",
                                "class": -1,
                                "content": "main characteristic"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    335,
                                    336
                                ],
                                "label": "",
                                "class": -1,
                                "content": "sample vineyard"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    620,
                                    621
                                ],
                                "label": "",
                                "class": -1,
                                "content": "education initiatives"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    612,
                                    613
                                ],
                                "label": "",
                                "class": -1,
                                "content": "research bundle"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    413,
                                    414
                                ],
                                "label": "",
                                "class": -1,
                                "content": "satellite image"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    451,
                                    452
                                ],
                                "label": "",
                                "class": -1,
                                "content": "drone image"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    442,
                                    443
                                ],
                                "label": "",
                                "class": -1,
                                "content": "sample area"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    210,
                                    211
                                ],
                                "label": "",
                                "class": -1,
                                "content": "ai model"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    587,
                                    588
                                ],
                                "label": "",
                                "class": -1,
                                "content": "ai community"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    163,
                                    164
                                ],
                                "label": "",
                                "class": -1,
                                "content": "ai4eu project"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    522,
                                    523
                                ],
                                "label": "",
                                "class": -1,
                                "content": "trustworthy ai"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    525,
                                    526
                                ],
                                "label": "",
                                "class": -1,
                                "content": "gdpr requirement"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    369,
                                    370
                                ],
                                "label": "",
                                "class": -1,
                                "content": "follow datum"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    475,
                                    476
                                ],
                                "label": "",
                                "class": -1,
                                "content": "meteorological datum"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    79
                                ],
                                "label": "",
                                "class": -1,
                                "content": "_"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    359,
                                    360
                                ],
                                "label": "",
                                "class": -1,
                                "content": "machine learning model"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    448,
                                    449
                                ],
                                "label": "",
                                "class": -1,
                                "content": "counting model"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    598,
                                    599
                                ],
                                "label": "",
                                "class": -1,
                                "content": "industrial verticals"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    601,
                                    602
                                ],
                                "label": "",
                                "class": -1,
                                "content": "open call"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    604,
                                    605
                                ],
                                "label": "",
                                "class": -1,
                                "content": "case studies"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    629,
                                    630
                                ],
                                "label": "",
                                "class": -1,
                                "content": "working groups"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    132,
                                    133
                                ],
                                "label": "",
                                "class": -1,
                                "content": "ai4agri-kg.zip develop"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    135,
                                    136
                                ],
                                "label": "",
                                "class": -1,
                                "content": "artificial intelligence team"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    140,
                                    141
                                ],
                                "label": "",
                                "class": -1,
                                "content": "kapodistrian university"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    159,
                                    160
                                ],
                                "label": "",
                                "class": -1,
                                "content": "ai4agricultural pilot"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    183,
                                    184
                                ],
                                "label": "",
                                "class": -1,
                                "content": "specific application"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    234,
                                    235
                                ],
                                "label": "",
                                "class": -1,
                                "content": "spatial dimension"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    264,
                                    265
                                ],
                                "label": "",
                                "class": -1,
                                "content": "ai4agriculture pilot"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    394,
                                    395
                                ],
                                "label": "",
                                "class": -1,
                                "content": "petiol analysis"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    399,
                                    400
                                ],
                                "label": "",
                                "class": -1,
                                "content": "soil metadata"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    542,
                                    543
                                ],
                                "label": "",
                                "class": -1,
                                "content": "legal notice"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    579,
                                    580
                                ],
                                "label": "",
                                "class": -1,
                                "content": "innovation program"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    402,
                                    403
                                ],
                                "label": "",
                                "class": -1,
                                "content": "grape maturity"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    435,
                                    436
                                ],
                                "label": "",
                                "class": -1,
                                "content": "grape detect"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    646,
                                    647
                                ],
                                "label": "",
                                "class": -1,
                                "content": "past event"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    331,
                                    332
                                ],
                                "label": "",
                                "class": -1,
                                "content": "specific parcel"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    366,
                                    367
                                ],
                                "label": "",
                                "class": -1,
                                "content": "kg contain"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    652
                                ],
                                "label": "",
                                "class": -1,
                                "content": "search"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    406
                                ],
                                "label": "",
                                "class": -1,
                                "content": "ndvi"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    459
                                ],
                                "label": "",
                                "class": -1,
                                "content": "datum"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    642
                                ],
                                "label": "",
                                "class": -1,
                                "content": "news"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    467
                                ],
                                "label": "",
                                "class": -1,
                                "content": "model"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    383
                                ],
                                "label": "",
                                "class": -1,
                                "content": "pilot"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    439
                                ],
                                "label": "",
                                "class": -1,
                                "content": "plant"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    329
                                ],
                                "label": "",
                                "class": -1,
                                "content": "grape"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    644
                                ],
                                "label": "",
                                "class": -1,
                                "content": "event"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    503
                                ],
                                "label": "",
                                "class": -1,
                                "content": "kg"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    389
                                ],
                                "label": "",
                                "class": -1,
                                "content": "parcel"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    557
                                ],
                                "label": "",
                                "class": -1,
                                "content": "ai4eu"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    590
                                ],
                                "label": "",
                                "class": -1,
                                "content": "organizations"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    592
                                ],
                                "label": "",
                                "class": -1,
                                "content": "projects"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    596
                                ],
                                "label": "",
                                "class": -1,
                                "content": "industry"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    627
                                ],
                                "label": "",
                                "class": -1,
                                "content": "ethic"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    632
                                ],
                                "label": "",
                                "class": -1,
                                "content": "workshop"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    634
                                ],
                                "label": "",
                                "class": -1,
                                "content": "osai"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    640
                                ],
                                "label": "",
                                "class": -1,
                                "content": "events"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    81
                                ],
                                "label": "",
                                "class": -1,
                                "content": "log"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    242
                                ],
                                "label": "",
                                "class": -1,
                                "content": "form"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    138
                                ],
                                "label": "",
                                "class": -1,
                                "content": "national"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    187
                                ],
                                "label": "",
                                "class": -1,
                                "content": "scope"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    455
                                ],
                                "label": "",
                                "class": -1,
                                "content": "yield"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    457
                                ],
                                "label": "",
                                "class": -1,
                                "content": "quality"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    397
                                ],
                                "label": "",
                                "class": -1,
                                "content": "production"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    222
                                ],
                                "label": "",
                                "class": -1,
                                "content": "resource"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    232
                                ],
                                "label": "",
                                "class": -1,
                                "content": "temporal"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    429
                                ],
                                "label": "",
                                "class": -1,
                                "content": "number"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    433
                                ],
                                "label": "",
                                "class": -1,
                                "content": "cluster"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    479
                                ],
                                "label": "",
                                "class": -1,
                                "content": "class"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    481
                                ],
                                "label": "",
                                "class": -1,
                                "content": "property"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    520
                                ],
                                "label": "",
                                "class": -1,
                                "content": "guideline"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    547
                                ],
                                "label": "",
                                "class": -1,
                                "content": "facebook"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    551
                                ],
                                "label": "",
                                "class": -1,
                                "content": "linkedin"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        },
                        {
                            "automatically_fill": "true",
                            "subject": {
                                "lemma_content": "",
                                "positions": [
                                    555
                                ],
                                "label": "",
                                "class": -1,
                                "content": "twitter"
                            },
                            "confidence": 0,
                            "property": {
                                "lemma_content": "rel:is_a",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "rel:is_a"
                            },
                            "weight": 0,
                            "value": {
                                "lemma_content": "keyword",
                                "positions": [
                                    -1
                                ],
                                "label": "",
                                "class": -1,
                                "content": "keyword"
                            },
                            "field_type": "keywords"
                        }
                    ],
                    "content": [
                        "Skip to main content Main navigation * AI Community * Organizations * Projects * Business & Industry * Industrial Verticals * Open Calls * Case Studies * Research * AI Catalog * Research Bundles * Education * Education Catalog * Education Initiatives * Education related news * Ethics * Working Groups * Workshop * OSAI * Services * News & Events * News * Events * Past Events * Web Caf\u00e9s * _ _ Search User account menu * _ _ Log in Menu Breadcrumb 1 .",
                        "Home _ _ 2 .",
                        "Research _ _ 3 .",
                        "AI Assets _ _ 4 .",
                        "AI4Agri Knowledge Graph AI4Agri Knowledge Graph A dataset from Earth Observation , Machine Learning models and vineyard data , in the form of a Knowledge Graph .",
                        "Dataset _ _ AI4Agri-KG.zip Developed by Artificial Intelligence Team - National and Kapodistrian University of Athens License Mozilla Public License 2.0 ( MPL-2.0 ) Main Characteristic This data set has been produced for the AI4Agricultural pilot in the AI4EU project , and contains all the data used and produced by that pilot .",
                        "This pilot is targeting a specific application in the scope of precision agriculture to help predicting the yield and assessing the quality of the production in vineyards using remote sensing data and AI models .",
                        "We have designed an OWL ontology to model the resources and then interlinked the different data both in the temporal and spatial dimensions .",
                        "The dataset contains in the form of a Knowledge Graph information coming from Earth Observation , Machine Learning models and vineyard data , that were used in the AI4Agriculture pilot .",
                        "The dataset is distributed in RDF N-Triples format .",
                        "We also provide the OWL ontology that was used to model the data .",
                        "Technical Categories Knowledge Representation Business Categories Agriculture Keywords Semantic web ArtificialIntelligence Agriculture Smart Farming Earth Observation Vineyards Last updated 24.11.2021 - 19:12 Detailed Description The Knowledge Graph ( KG ) contains information from various sources regarding different attributes of grapes in specific parcels of the sample vineyards .",
                        "Based on the geospatial information of each sample area , we were able to link information coming from Earth Observation , Machine Learning models and vineyard data .",
                        "The KG contains the following data for the years 2020 and 2021 , that were used in the pilot : * Vineyards data ( parcels , sample areas , petiol analysis , production , soil metadata , grape maturity ) * NDVI ( corrected NDVI values based on Satellite images ) * Drone Images ( geolocated plant images from the sampling areas ) * Number of Clusters ( clusters of grapes detected in the plants of the sample areas , based on a counting model and drone images ) * Yield and Quality ( data for each sampling area calculated by a model that uses the above sources along with meteorological data ) The classes and properties that were created to model and link the data are available in the OWL ontology that is provided along with the KG ( RDF N-Triples format ) .",
                        "Trustworthy AI The AI4Agri Knowledge Graph does not violate the guidelines for Trustworthy AI .",
                        "GDPR Requirements The AI4Agri Knowledge Graph itself is GDPR compliant .",
                        "Image Footer * About AI4EU * Legal notice * ai4eu in facebook * ai4eu on linkedin * ai4eu on twitter * ai4eu on youtube EU Footer Image European flag This project has received funding from the European Union 's Horizon 2020 research and innovation program under grant agreement 825619 .",
                        "* AI Community * Organizations * Projects * Business & Industry * Industrial Verticals * Open Calls * Case Studies * Research * AI Catalog * Research Bundles * Education * Education Catalog * Education Initiatives * Education related news * Ethics * Working Groups * Workshop * OSAI * Services * News & Events * News * Events * Past Events * Web Caf\u00e9s * Search"
                    ],
                    "data_source": "converter-service",
                    "source_doc_id": "https:\/\/www.ai4europe.eu\/research\/ai-catalog\/ai4agri-knowledge-graph"
                },
                "index-name": "default-text-index",
                "short-answers": [],
                "_score": 1.0,
                "_index": "default-text-index",
                "_id": "cacheid_ba28d93c4f7a050e388be710becee447",
                "_type": "_doc",
                "highlight": []
            }
        ],
        "max_score": 1.0,
        "total_docs": 1,
        "query": {
            "_source": {
                "excludes": [
                    "lemma_title",
                    "sentiment",
                    "text_suggester",
                    "lemma_content"
                ]
            },
            "from": 0,
            "size": 1,
            "query": {
                "bool": {
                    "must": [
                        {
                            "bool": {
                                "must": [
                                    {
                                        "wildcard": {
                                            "source_doc_id": "https:\/\/www.ai4europe.eu\/research\/ai-catalog*"
                                        }
                                    },
                                    {
                                        "match": {
                                            "content": "dataset"
                                        }
                                    },
                                    {
                                        "match": {
                                            "content": "agriculture"
                                        }
                                    },
                                    {
                                        "match": {
                                            "content": "knowledge representation"
                                        }
                                    }
                                ]
                            }
                        },
                        {
                            "bool": {
                                "should": [
                                    {
                                        "match": {
                                            "content": {
                                                "query": "deep learning resources"
                                            }
                                        }
                                    },
                                    {
                                        "match": {
                                            "content": {
                                                "query": "deep learning resources",
                                                "operator": "and"
                                            }
                                        }
                                    },
                                    {
                                        "match": {
                                            "lemma_content": {
                                                "query": "deep learne resource"
                                            }
                                        }
                                    },
                                    {
                                        "match": {
                                            "lemma_content": {
                                                "query": "deep learne resource",
                                                "operator": "and"
                                            }
                                        }
                                    },
                                    {
                                        "match": {
                                            "title": {
                                                "query": "deep learning resources"
                                            }
                                        }
                                    },
                                    {
                                        "match": {
                                            "title": {
                                                "query": "deep learning resources",
                                                "operator": "and"
                                            }
                                        }
                                    },
                                    {
                                        "match": {
                                            "lemma_title": {
                                                "query": "deep learne resource"
                                            }
                                        }
                                    },
                                    {
                                        "match": {
                                            "lemma_title": {
                                                "query": "deep learne resource",
                                                "operator": "and"
                                            }
                                        }
                                    },
                                    {
                                        "nested": {
                                            "path": "kg",
                                            "query": {
                                                "bool": {
                                                    "must": [
                                                        {
                                                            "nested": {
                                                                "path": "kg.value",
                                                                "query": {
                                                                    "match": {
                                                                        "kg.value.lemma_content": "keyword"
                                                                    }
                                                                }
                                                            }
                                                        },
                                                        {
                                                            "nested": {
                                                                "path": "kg.subject",
                                                                "query": {
                                                                    "match_phrase": {
                                                                        "kg.subject.content": {
                                                                            "query": "deep learne resource",
                                                                            "boost": 0.5,
                                                                            "slop": 1
                                                                        }
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    ]
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            }
        }
    },
    "info": "searching-LK526RWpuoOk99THO7KInqcjpQVbLE3Y-1-9-MainThread",
    "config": "\/home\/searchai_svc\/searchai\/app\/projects\/default\/configs\/search.json",
    "version": "1.0.0",
    "date": "2021\/03"
}


search_item = SearchItemInFocus(test.get('results').get('items')[0])

print(search_item.get_keywords())
search_item.print()
print(search_item.get_keywords())
'''