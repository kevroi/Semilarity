# Semilarity
## Overview
A natural language processing program that collects BBC news articles and uses a neural network to calculate the semantic similarities and distances to place concepts in an abstract metric space from the perspectives of the writers' semantics and search engine's scope.

## Usage
As of the last commit, the python script containing the entire program is stored in `main.py`.
`keywords.xlsx` stores the ten keywords that are input into the program. Support for a variable number of keywords will be added soon.

## Architecture
![Architecture Schematic for Semilarity](./images/SemilarityArch.png)

## How it Works

## Results
For the deault set of search terms (which are cyber security related, and found in `keywords.xlsx` in this repo), the following semantic distances were obtained,  
![Heatmap of Semantic Distances](./images/SemDist_heatmap.png)  

with the semantic distances and cosine similarity scores being distributed as follows.  
![Histogram of Semantic Distances and Cosine Similarities](./images/Sem_histograms.png)  

To get an idea of where each search term lies with respect to eachother, the violin plot shows a kernel density estimate of the spread of typical distances for each search term.  

![Violin plot of distances](./images/SemDist_violin.png)  

Additionally, the Term Frequency Inverse Document Frequency (TF-IDF) statistic was also used to compare the interpretations of the two algorithms to the same corpus.

![Heatmap of TFIDF](./images/TFIDF_heatmap.png)  


## Dependencies
The full scope of interaction of dependencies was described in the Architecture section. The dependencies used in this project are:  

,
as listed in `requirements.txt`.

## Installation