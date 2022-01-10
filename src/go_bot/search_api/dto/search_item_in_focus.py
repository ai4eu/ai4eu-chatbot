# SearchItemInFocus is a dictionary
# Get only what we need from the search-API response
# Latest Search API json format
class SearchItemInFocus:
    def __init__(self, response:dict):
        source = response.get('_source')
        self.score = response.get('_score')
        self.index = response.get('_index')
        self.id = response.get('_id')
        self.type = response.get('_type')
        self.url = source.get('_source_doc_id')
        self.content = source.get('content')
        self.title = source.get('title')

        # Get summary - is an array of summaries
        summaries = source.get('summary')
        if summaries is None or len(summaries) is 0:
            self.summary = 'Unfortunately there is no summary for this resource'
        else:
            # get the first one
            self.summary = source.get('summary')[0]

        # Get keywords from categories
        self.keywords = self._get_top_keywords(source.get('category'))

    @classmethod
    def with_validation(cls, response:dict):
        # Check that there is a source
        source = response.get('_source')

        if source is None:
            return None

        search_item = cls(response)

        return search_item

    """
    category is an array of dicts with labels and scores
    parse them all and get the top-3
    """
    @classmethod
    def _get_top_keywords(cls, category):
        if category is None:
            return []
        # Holds the keywords and their aggregated scores
        keywords = {}
        # For all sets of keywords and scores
        for set_of_keywords in category:
            # For all keywords
            for i, keyword in enumerate(set_of_keywords.get('labels')):
                if keyword in keywords.keys():
                    score = keywords.get(keyword)
                    # Add the score of index i
                    score += set_of_keywords.get('scores')[i]
                    keywords[keyword] = score
                else:
                    # Just add the value
                    score = set_of_keywords.get('scores')[i]
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
test = [{
                    "labels": [
                        "machine learning",
                        "natural language processing",
                        "deep learning",
                        "reinforcement learning",
                        "robotic",
                        "neuromorphic computing",
                        "internet of things",
                        "computer vision"
                    ],
                    "scores": [
                        0.226396968588233,
                        0.16912697907537222,
                        0.24667231924831867,
                        0.1327694384381175,
                        0.08736297395080328,
                        0.03549416735768318,
                        0.06415985617786646,
                        0.03801731951534748
                    ]
                },
                {
                    "labels": [
                        "natural language processing",
                        "machine learning",
                        "deep learning",
                        "reinforcement learning",
                        "robotic",
                        "internet of things",
                        "neuromorphic computing",
                        "computer vision"
                    ],
                    "scores": [
                        0.43566924426704645,
                        0.16076652891933918,
                        0.1865750551223755,
                        0.0873623676598072,
                        0.03752412088215351,
                        0.04472467303276062,
                        0.016063742339611053,
                        0.031314258463680744
                    ]
                }
            ]


print(SearchItemInFocus.get_top_keywords(test))
'''