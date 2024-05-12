
# README for President-NLP-Entities-Relations Notebook

## Data
got it from https://data.millercenter.org/ , down loaded as a json

## Overview
This Jupyter notebook focuses on Natural Language Processing (NLP), specifically on extracting, processing, and analyzing data related to entities and their relations found in US Presidential speeches. It uses a variety of Python libraries such as: 

-seaborn
-Counter
-datetime
-MinMaxScaler
-StandardScaler
-TextBlob
-json
-pandas
-state_union
-spacy
-pandas
-neuralcoref
-networkx
-matplotlib.pyplot

## Setup and Requirements
- Python environment (preferably Anaconda for ease of package management).
- Libraries required: `spacy`, `seaborn` (for visualization), and others as specified in the notebook.
- The notebook mentions data stored in CSV format, so ensure relevant data files are accessible.
- Yet to use the portions that require neuralcoref spacey 2.1.1 is needed and cant be pip installed in Jypyter so a self made conda environment is advised to run the neural_coref.py file, and the same goes for the relation.py file as well, yet i ahve already run it and saved the data

## Usage
- Follow the instructions and comments within the notebook for step-by-step guidance.
- Some code sections are commented out due to long execution times. They can be uncommented and run if needed.
- The notebook includes data visualization, so interpret the graphs and charts as per the context provided in the notebook.

## Additional Information
- The notebook references external JSON data sources; ensure you have the correct data files.
- It's designed for educational and analytical purposes in the field of NLP.
- multiple csv fils come in the folder that the ipynb file uses as bench marks to avoid using the code that hasa long execution time
- additional python files that can be run seperately from the main ipynb file to make use of the terminal instead.
