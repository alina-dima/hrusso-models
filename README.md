# Next-Word Prediction in Low-Resource Languages: A Study on Hrusso Aka
### Master's Research Project
#### By: Alina Dima
#### Supervised by: Dr. Stephen Jones, Dr. Tsegaye Tashu

**Acknowledgements**: Many thanks to Dr. Vijay D'Souza and NEILAC for providing the dataset used to train the models and additional resources regarding the Hrusso Aka language.

### Abstract
Add abstract

## General Description
The current repository contains the code associated with the NLP models developed for my thesis (add thesis pdf link when available) in the context of the (very) low-resource language Hrusso Aka. Additionally, we included the code associated with preprocessing, data exploration, and a trial for developing word embeddings for Hrusso Aka.

Thus, this repository can serve as comprehensive inspiration for anyone who wishes to contribute to NLP for (very) low-resource languages.

For the keyboard implementation and the next-word prediction model integration see https://github.com/alina-dima/hrusso-keyboard.

## Usage Guidelines:
The repository is split into multiple folders depending on their purpose, each with their associated code.

If one wishes to implement an end-to-end next-word prediction model for a (very) low-resource language, then we recommend following the code in the order below, where the models are ordered based on complexity. 

- **`data_preprocessing`**: Ensure that your data is ready for training the models. In our case, preprocessing did not involve many steps due to the dataset being rather clean (proper spellings, no punctuation, etc.).
  - This folder contains a Jupyter notebook used to preprocess the raw data extracted from the ELAN audio files. Additionally, it contains another Jupyter notebook meant to provide a landscape of the main topics present in the dataset. This is achieved using KeyBERT, which extracts the main keywords from the English translation of the files.

- **`embedding_models`**: You could also experiment with whether embedding models could be successfully trained. If this is the case, you can explore further options for next-word prediction model training compared to this thesis. In our case, for Hrusso Aka, the embedding models did not perform well in our evaluation, so we proceeded without pre-learned embeddings and allowed the more complex models to learn their own embeddings.

- **`ngrams`**: This is the least complex model implemented, where we explored different n-gram orders and smoothing techniques. For further information, refer to the README file present in that folder.

- **`rnn`**: In this case we compared and optimised LSTM- and GRU-based recurrent neural networks. For further information, refer to the README file present in that folder.

- **`transformer`**: The most complex of the models, a decoder-only transformer model. For further information, refer to the README file present in that folder.

