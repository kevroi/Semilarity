# Tested on the following libraries:
# python version 3.8.5
# beautifulsoup4 version 4.9.3
# gensim version 4.0.1
# matplotlib version 3.4.1
# networkx version 2.5.1
# nltk version 3.6.1
# numpy version 1.20.2
# openpyxl version 3.0.6
# pandas version 1.2.3
# requests version 2.25.1
# scikit-learn version 0.24.1
# scipy version 1.6.2
# seaborn version 0.11.1
# sklearn version 0.0
# soupsieve version 2.2

import openpyxl
import requests
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup as bs
from bs4.element import Comment
import os
import time
import re
import numpy as np

start_time = time.time()


# Download webpage content from relevant tags
def relevant_tag(element):

    #span gets rid of the 
    if element.parent.name in unwanted_tags:
        return False
    if isinstance(element, Comment):
        return False
    
    return True


# Use beautiful soup to collect & process article contents
def html_to_text(body):

    soup = bs(body, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(relevant_tag, texts)
    result = u" ".join(t.strip() for t in visible_texts)
    # get rid of share copy link buttons' text
    result = result.replace("Share page Copy link ", "")
    # get rid of subheading texts at bottom of page
    result = result.rsplit("Published Published")[0]

    return result


# Save article contents to reasonable data structure
def save_to_txt(string):

    global keyword, article_counter
    script_dir = os.path.dirname(os.path.realpath(__file__))
    rel_path = f'/{keyword}/Article{article_counter}.txt'
    abs_file_path = script_dir + rel_path
    txtfile = open(abs_file_path, 'wt', encoding='UTF-8')
    txtfile.write(string)
    txtfile.close()



# location of keywords file
path = 'keywords.xlsx'
# create workbook object
book = openpyxl.load_workbook(path)
# select relevant sheet
sheet = book['Sheet1']
kw = sheet['A2':'A11']
kwlist = [i[0].value for i in kw]


# access BBC webpage
url = 'https://www.bbc.co.uk'

# links that appear on all search result pages that we DO NOT
# want to extract text from

unwanted_links = ['https://www.bbc.co.uk/news/localnews',
                'https://www.bbc.co.uk/news/help-41670342']

# HTML tags on the results page that we DO NOT want to scrape
unwanted_tags = ['style', 'script', 'head', 'title', 'meta', '[document]',
                'span', 'h2', 'figcaption', 'path', 'a']

# Maximum articles to scrape from a webpage
max_articles = 100

# create a list data structure to store results from each keyword.
corpus = [""]*len(kwlist)

for keyword in kwlist:

    article_counter = 0

    # Downloading webpage content
    # Create folder to store articles
    if not os.path.exists(keyword):
        os.makedirs(f'{keyword}')

    for pageNumber in range(1, 30):
        values = {'q' : keyword, 'page' : pageNumber}
        data = urllib.parse.urlencode(values)
        
        try:
            r = requests.get(url + '/search?' + data)
        except:
            print("BBC prevented program from accessing search page. Retrying...")
            while r.status_code != 200:
                try:
                    time.sleep(2)  
                    r = requests.get(url + '/search?' + data)
                    print("Success!")
                except:
                    pass

        soup = bs(r.content, 'html.parser')
        
        #find all links on this result page that leads to news articles, remove unwanted links
        titles_html = soup.findAll("a", {"href" : lambda L: L and 
                                        (L.startswith('https://www.bbc.co.uk/news/') or L.startswith('http://www.bbc.co.uk/news/')
                                         or L.startswith('https://news.bbc.co.uk/') or L.startswith('http://news.bbc.co.uk/'))
                                        and (L not in unwanted_links)}
                                    )

        for i in titles_html:
            
            # download html
            try:
                html_response = urllib.request.urlopen(i.get('href'))
            except:
                print(f"BBC prevented program from accessing an article for {keyword}. Retrying...")
                while html_response.getcode() != 200:
                    try:
                        time.sleep(2) 
                        html_response = urllib.request.urlopen(i.get('href'))
                        print("Success!")
                    except:
                        pass

            article_string = html_to_text(html_response.read())

            # Collection and processing to a data structure
            # Need to remove characters from string in case some machines can not handle
            article_string = " ".join(re.sub(r'[^a-zA-Z]','',w).lower() for w in article_string.split())

            corpus[kwlist.index(keyword)] = corpus[kwlist.index(keyword)] + article_string + " "
    
            article_counter += 1

            # save article as txt file
            try:
                save_to_txt(article_string)

            except:
                print(f"Article {article_counter}, for the keyword '{i.text}' contains characters not in your machine's character set.")
                print(f"Article {article_counter} will still be processed via python data structures for Problem 3")

            print(f'{article_counter} articles downloaded for "{keyword}" (keyword {kwlist.index(keyword)+1}/{len(kwlist)})', end='\r')

            if article_counter >= max_articles:
                break

        if article_counter >= max_articles:
            break

    print("")




# TF-IDF Implementation
from sklearn.feature_extraction.text import TfidfVectorizer
print("Vectorizing articles...")
vectors = TfidfVectorizer(min_df=1, stop_words="english")
tfidf_fit = vectors.fit_transform(corpus)
tfidf_similarity = tfidf_fit * tfidf_fit.T
tfidf_matrix = tfidf_similarity.toarray()

# Doc2Vec Implementation
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

# load articles into dataframe
articles_df=pd.DataFrame(corpus,columns=['documents'])

# removing special characters and stop words from the text
print("Downloading Stopwords...")
nltk.download('stopwords')
stop_words_l=stopwords.words('english')
# clean document and regex substitute all strings beginning with a non-alphabet
articles_df['documents_cleaned']=articles_df['documents'].apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]','',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]','',w).lower() not in stop_words_l) )

# downloading punkt
print("Downloading Punkt tokenizer from Natural Language Toolkit...")
nltk.download('punkt')

#tokenize articles
print("Tokenizing news articles to tagged data...")
tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(articles_df['documents_cleaned'])]
model_d2v = Doc2Vec(vector_size=100,alpha=0.025, min_count=1)

model_d2v.build_vocab(tagged_data)

# Train model by updating its neural weights
# use 100 epochs
print("Training Neural Network...")
for epoch in range(100):
    model_d2v.train(tagged_data,
                total_examples=model_d2v.corpus_count,
                epochs=model_d2v.epochs)
    

print("Embedding documents into vector space...")
# document_embeddings will contain the 10 document vectors
# in this 100 dimensional space that are isomorphic to the
# 10 keywords
document_embeddings=np.zeros((articles_df.shape[0],100))

for i in range(len(document_embeddings)):
    document_embeddings[i]=model_d2v.dv[i]
   
# Calculate cosine similarity and distance between keywords
# into 2D numpy arrays

print("Calculating pairwise similarities between documents...")
similarity_matrix=cosine_similarity(document_embeddings)

# Do the same for distance_matrix
print("Calculating pairwise distances between documents...")
distance_matrix = euclidean_distances(document_embeddings)

# Export Distances to spreadsheet
print("Exporting distance matrix to a spreadsheet (distance.xlsx)...")
DistMat_df = pd.DataFrame(distance_matrix, index=kwlist, columns=kwlist)
# setting index to true below ensures row index from above is carreid over.
DistMat_df.to_excel("distance.xlsx", index=True)

finish_time = time.time() - start_time
print(f"\n Program completed in {finish_time // 60} minutes {finish_time % 60} seconds")


# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

sns.set_theme()
title_style = {
    'fontsize': 18,
    'fontweight': 'bold'
}

# Create a Heatmap of distances between all keywords
ax1 = sns.heatmap(distance_matrix, annot=True, xticklabels=kwlist, yticklabels=kwlist)
ax1.set_title("Semantic distance heatmap for all keywords", fontdict=title_style)
plt.show()


# Create Histograms of distance and similarity distribution
# Flatten arrays and remove diagonal entries first
flat_DistMat = [item for elem in distance_matrix for item in elem]
flat_DistMat = np.array(flat_DistMat)
flat_DistMat = np.where(flat_DistMat == 0.0, np.nan, flat_DistMat)

SimMat_NoDiag = np.copy(similarity_matrix)
np.fill_diagonal(SimMat_NoDiag, np.nan)
flat_SimMat = [item for elem in SimMat_NoDiag for item in elem]
flat_SimMat = np.array(flat_SimMat)

ax2 = sns.histplot(data=flat_DistMat, bins=10)
ax2.set_title("Distance distribution", fontdict=title_style)
ax2.set(xlabel="Semantic Distance")
plt.show()

ax3 = sns.histplot(data=flat_SimMat, bins=10)
ax3.set_title("Similarity distribution", fontdict=title_style)
ax3.set(xlabel="Cosine Similarity")
plt.show()


# Create a graph visualising semantic positions of keywords
graph = nx.from_numpy_matrix(distance_matrix)
G = nx.relabel_nodes(graph, dict(enumerate(DistMat_df.columns)))
nx.draw(G, with_labels=True, font_size=8, width=0.5, node_color='skyblue',
        edge_color='gray')
plt.show()


# Create a violin plot with distances from each keyword
violin_df = pd.DataFrame(columns=["Keyword", "Distance to other keywords"])

for keyword in kwlist:
    for value in DistMat_df[keyword]:
        if value == 0:
            pass
        else:
            violin_df.loc[len(violin_df.index)] = [keyword, value]

ax4 = sns.violinplot(x="Keyword", y="Distance to other keywords", data=violin_df,
                    scale="area")
ax4.set_title("Spread of semantic distances for each keyword", fontdict=title_style)
ax4.set_xticklabels(kwlist, size = 8)
plt.show()

ax5 = sns.heatmap(tfidf_matrix, annot=True, xticklabels=kwlist, yticklabels=kwlist,
                    cmap=sns.cm.rocket_r)
ax5.set_title("TF-IDF score heatmap for all keywords", fontdict=title_style)
plt.show()