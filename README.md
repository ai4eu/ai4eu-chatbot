# ai4eu-chatbot
## A user friendly chatbot for the AI4EU platform

The ai4eu-chatbot is a chatbot developed by [FORTH-ICS](https://www.ics.forth.gr/) for the [AI4EU](https://www.ai4europe.eu/) project.  The chatbot was developed on top of the [deeppavlov](https://deeppavlov.ai/) open-source conversational AI framework and it is a multi-task goal oriented chatbot.

## Features
The chatbot is provided as a REST service and supports the following tasks:
- Question-Anwser (QA)
-- Domain specific QA about the AI4EU project and general AI questions (e.g., "what is AI4EU?", "what are transformers?" )
-- Open-domain QA using the pretrained deeppavlov wikidata KBQA model (e.g., "who is Seymour Cray?")
- Exploration of web resources indexed by the AI4EU platform
-- Uses the Search-API of the AI4EU project
- Exploration of the available assets in the [AI4EU Asset Catalogue](https://www.ai4europe.eu/research/ai-catalog)
-- Uses the Search-API of the AI4EU project

## Architecture

## Deep learning models used

- Sentence Embeddings
   Uses the [bert-sentence_multi_cased_L-12_H-768_A-12_pt](https://github.com/deepmipt/DeepPavlov/blob/master/deeppavlov/configs/embedder/bert_sentence_embedder.json) model provided by deeppavlov. We just use the pretrained model so there is no need for training

- Slot-filler just uses a fuzzy slot mapping approach

- Question-Answering
    Uses a fine-tuned model over the sentence embeddings [Microsoft mpnet-base](https://huggingface.co/microsoft/mpnet-base) model. The model configuration is provided at [config/qa/sentence-emb/all-mpnet-base-v2.json](https://github.com/ai4eu/ai4eu-chatbot/blob/main/config/qa/sentence-emb/all-mpnet-base-v2.json)

- Intents Classifier
   Uses a fine-tuned model over the sentence embeddings [Microsoft mpnet-base](https://huggingface.co/microsoft/mpnet-base) model. The model configuration is provided at [config/intents/sentence-emb/all-mpnet-base-v2.json](https://github.com/ai4eu/ai4eu-chatbot/blob/main/config/intents/sentence-emb/all-mpnet-base-v2.json)

For more information about sentence-embedding models and their performance, visit https://www.sbert.net/docs/pretrained_models.html
## Installation
Initially create a conda environment using the following command:
```sh
conda create -n "ai4eu-chat" python=3.7.10
```
Then enable the conda environment by running
```sh
source ./scripts/conda-activate.sh
```

In order to run the chatbot you will need the following packages

- deeppavlov
    We need version 0.17.0
    ```sh
        pip install deeppavlov==0.17.1
    ```

- pytorch
    We need torch for the models. Since the development was done on an AMD Radeon VII gpu we used the rocm enabled pip package which can be downloaded from https://download.pytorch.org/whl/rocm4.0.1/torch-1.9.0%2Brocm4.0.1-cp37-cp37m-linux_x86_64.whl
    ```sh
        pip install torch-1.9.0+rocm4.0.1-cp37-cp37m-linux_x86_64.whl
    ```
- tensorflow
    We also need tensorflow 1.15 for some components. For RTX 30x cards please use the following for tensorflow (needs though a python 3.8 environment)
    ```sh
        pip install nvidia-pyindex
        pip install nvidia-tensorflow[horovod]
        pip install nvidia-tensorboard==1.15
    ```
- hugging face libraries for transformers transformers (4.6.0) and sentence-transformers (2.0.0)
    ```sh
    pip install transformers==4.6.0
    pip install sentence-transformers==2.0.0
    ```
- sanic
    The REST service is provided using sanic
    ```sh
        pip3 install sanic
    ```

#### Building and installing the models
Before running the chatbot you will need to train the various models over the datasets that are provided in the repository
- QA model
    ```sh
       python -m deeppavlov train  config/qa/sentence-emb/all-mpnet-base-v2.json
    ```
- Intents model
    ```sh
       python -m deeppavlov train  config/intents/sentence-emb/all-mpnet-base-v2.json
    ```
- goal-oriented bot model
    ```sh
       python -m deeppavlov train  config/gobot/ai4eu-gobot.json
    ```

You can interact with each model using the following command
```sh
python -m deeppavlov train PATH_TO_MODEL_CONFIGURATION
python -m deeppavlov evaluate PATH_TO_MODEL_CONFIGURATION
python -m deeppavlov interact PATH_TO_MODEL_CONFIGURATION
```

# How to run
To start the service at port 8000 please run:
```sh
nohup python src/REST_Chatbot.py &
```
# A simple request
The service is currently deployed at FORTH's premises. You can make queries to the chatbot using the following command:
```sh
    curl -X POST "https://pangaia.ics.forth.gr/ai4eu-chatbot/" -d "{\"query\":\"What is AI4EU?\"}"
```

In addition you will need to install the wikidata kbqa_cq model and the bert sentence embeddigns model using the following commands
```sh
    python -m deeppavlov install kbqa_cq -d
    python -m deeppavlvo install bert_sentence_embedder -d
```



Add also -d to the above when running for the first time, so that it will  download any needed pretrained models

## Development

The chatbot was developed by [Panagiotis Papadakos](mailto:papadako@ics.forth.gr). Please contact him for any questions.

## License

Original deeppavlov code under Apache License, Version 2.0
FORTH's contributions are under EUPL

