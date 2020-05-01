import os
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy import displacy
from sklearn import svm
from joblib import load
import model


def classify(data):
    if not os.path.isfile('model.joblib'):
        model.trainModel()

    clf = load('model.joblib')
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



