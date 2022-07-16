# This project is still under development, hence many functionality may be incorrect or partially done in our public repo.
# Project: Like a bilingual baby: The advantage of visually grounding a bilingual language model

## Project Guide:
1. Download the MS-COCO-ES dataset.
2. Put all images in the image/ directory
3. Run the code

## Project structure:

Files:

- multi_bpe.py: Contains the MultiBPE class, which allows user to load the pretrained tokenizer and embedding matrix 

- image_caption_dataset.py: Contains the torch's dataset wrapper

- multimodal_model.py: Contains the baseline LSTM and multimodal LSTM language models

- multimodal_lstm.py: Contains the multimodal LSTM

Training Notebooks:

- train_visual.ipynb: Run this to train and save the multimodal LSTM

- train_benchmark.ipynb: Run this to train and save the benchmark LSTM

Testing Notebooks:

- rg65-en_es.ipynb: Used to test the similarity score of the RG-65_EN-ES dataset

- rg65-es.ipynb: Used to test the similarity score of the RG-65_ES dataset

- rg65.ipynb: Used to test the similarity score of the RG-65 dataset

- simlex-999.ipynb: Used to test the similarity score of the SimLex999 dataset

- wordsim_relatedness_goldstandard.ipynb: Used to test the similarity score of the WordSim-R dataset

- wordsim_similarity_goldstandard.ipynb: Used to test the similarity score of the WordSim-S dataset

- wordsim353_agreed.ipynb: Used to test the similarity score of the WordSim353 dataset

- men.ipynb: Used to test the similarity score of the MEN dataset

Directories:

- data: contains all dataset used for training and testing

- images: contains some test images and all the images in the MS-COCO-ES dataset for training


## MSCOCO-ES Dataset:

- Download at https://github.com/carlosgarciahe/ms-coco-es