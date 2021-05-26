from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model
from deeppavlov import configs, train_model

import numpy as np

# Load the faq model using the distilbert base sentence embedding
faq_sentence_emb = build_model('config/faq/sentence-emb/distilbert-base-nli-stsb-mean-tokens.json')

# A simple question
result = faq_sentence_emb.compute(['When calls will be launched?'], targets=['y_pred_labels', 'y_pred_probas'])

# Get the label and its probability
label = result[0][0]
prob = np.amax(result[1])

# Print the result
print('label: ' + label + ' probability:' + str(prob))
