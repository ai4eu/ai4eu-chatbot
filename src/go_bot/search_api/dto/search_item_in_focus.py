# SearchItemInFocus is a dictionary
# Get only what we need from the search-API response
# Latest Search API json format
class SearchItemInFocus:
    def __init__(self, response: dict):
        source = response.get('_source')
        self.score = response.get('_score')
        self.index = response.get('_index')
        self.id = response.get('_id')
        self.type = response.get('_type')

        if source.get('source_doc_id'):
            self.url = source.get('source_doc_id')
        else:
            self.url = 'Unknown'

        if source.get('content'):
            self.content = source.get('content')
        else:
            self.content = 'Unknown'

        # Get summary - is an array of summaries
        summaries = source.get('summary')
        if summaries is None or len(summaries) is 0:
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

        print(filtered_categories)

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
'''
test = {
    "results": {
        "items": [
            {
                "_source": {
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
                    }
                }
            }
        ]
    }
}


search_item = SearchItemInFocus(test.get('results').get('items')[0])

print(search_item.get_keywords())
'''