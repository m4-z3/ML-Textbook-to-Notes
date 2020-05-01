from lda import TopicModeling
from sent_class import classify
import sys
import os

def main():
    # initialize lda stuff
    lda_modeling = TopicModeling()
    # get file contents
    with open(sys.argv[1], 'r', encoding='utf-8') as txt:
        paragraphs = txt.readlines()
    
    output = []

    # go through all paragraphs
    for paragraph in paragraphs:
        groupedSentence = lda_modeling.groupSentence(paragraph) 
        output += classify(groupedSentence)

    # write to output file
    with open('Notes.txt', 'w') as file:
        file.writelines(output)

if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1][-4:] != '.txt':
        sys.exit('Please pass in a txt file')
    elif len(sys.argv) > 2:
        sys.exit('Too many arguments passed in')
    if os.path.isfile(sys.argv[1]):
        main()
    else:
        sys.exit("File doesn't exist")