# This project is still under development, hence many functionality may be incorrect or partially done in our public repo.
# Project: Understanding Bilingualism with the Multimodal DeltaRNN model

## Project Update:


## Project structure:

Files:

- multi_bpe.py: Contains the MultiBPE class, which allows user to load the pretrained tokenizer and embedding matrix 

- image_caption_dataset.py: Contains the torch's dataset wrapper

- multimodal_model.py: Contains the baseline LSTM and multimodal LSTM language models

- multimodal_lstm.py: Contains the multimodal LSTM
 

Directories:

- data: contains all dataset used for training and testing

- images: contains some test images

- wordsim_exps: contains all notebooks used to test word similarity score


## MSCOCO-ES Dataset:

- Download at https://github.com/carlosgarciahe/ms-coco-es