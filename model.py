import csv
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy import displacy
from sklearn import svm
from joblib import dump, load



with open('data.csv', newline='', encoding='utf-8') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    header = next(readCSV)
    rows = [[row[0], int(row[1]), int(row[2])] for row in readCSV]
    ary = []
    len_sent = []
    for row in rows:
        doc = nlp(row[0])
        len_sent.append(len(row[0]))
        ary.append(len(doc.ents))

    x = list(zip(len_sent[0:20], ary[0:20]))
    y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    clf = svm.SVC(kernel= 'linear', C= 1.0)
    clf.fit(x,y)



    dump(clf, "model.joblib")

