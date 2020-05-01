import csv
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy import displacy
import matplotlib.pyplot as plt
import numpy as np
import scipy, pylab
from sklearn.cluster import KMeans
from sklearn import svm
from matplotlib import style
style.use("ggplot")
from lda import TopicModeling
from joblib import dump, load

clf = load('model.joblib') 

def classify(data): 

    ##opens Notes.txt file 
    file1 = open("Notes.txt","a")
    ## goes through and gets the entities and size
    ## needed for the prediction 
    for i in range(len(data)):
        entities = []
        total_length = []
        predict_array = []
        ## creates array of predictions for each sentence 
        ## in a topic in each paragraph 
        for j in range(len(data[i])): 
            doc = nlp(data[i][j])
            entities.append((len(doc.ents)))
            total_length.append((len(data[i][j])))
            predictions = list(zip(total_length, entities))
            predict_array = clf.predict(predictions)
        
        file1.write("-" + data[i][0])
        file1.write("\n")
        ## goes through and indents how the sentences are 
        ## based on the calculated prediction
        for y in range(1,len(predict_array)): 
            if(predict_array[y] == 1 and y != 0): 
                file1.write("\t" + "\t" + "-" + data[i][y])
                file1.write("\n")
            elif(predict_array[y] == 0 and y != 1):
                file1.write("\t" + "-" + data[i][y])
                file1.write("\n")
    file1.write("\n")

def sentClassify():
    lda_modeling = TopicModeling()
    ##gathers paragraphs from textbook 
    with open('Ch1-HumanGeo.txt', 'r', encoding='utf-8') as txt:
        paragraphs = txt.readlines()
    ##groups the sentences using lda topic modeling class 
    for paragraph in paragraphs[0:60]:
        groupedSentence = lda_modeling.groupSentence(paragraph)    
        ##calls classify function for each paragraph  
        classify(groupedSentence)
        
def main():
    alphaValues = ['symmetric', 'asymmetric', 'auto', 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    etaValues=[None, 'auto', 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    sentClassify()

main()


