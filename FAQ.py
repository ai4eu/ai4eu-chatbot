from deeppavlov import configs
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model
from deeppavlov import configs, train_model

model_config = read_json('config/faq/ai4eu_faq_transformers.json')

faq = build_model('config/faq/ai4eu_faq_transformers.json')

result = faq(['What calls are available?'])

print(result)
