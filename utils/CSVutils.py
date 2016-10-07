'''
Created on Sep 29, 2016

@author: daniel
'''

import csv
import os

'''
 Read a CSV file and return 
 delimiter defines the separation character between columns in the CSV file
 Returns an array of tuples that contains the data and its label
'''
def readCSV(filename, delimiter=','):
    out = []
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
        for row in spamreader:
            # For reading csv with photo names
            out += [(row[0], int(row[1]), row[2])]
    return out

'''
 Update the photo list csv with labels
'''
def updateCSVFile(filePath):
    # Runs shell script to update CSV with photo names, id and person name
    command = "{}update_csv.sh".format(filePath)
    os.system(command)
    print filePath

'''
 Gets the array from the readCSV function and returns two different arrays:
   dataArray - contains the data of the file, preprocessed face or pretrained data 
   labelArray - contains the name of the person to which this data belongs to.
'''
def getPhotoAndLabel(filename):
    parsedCsv = readCSV(filename)
    dataArray, labelArray, translationArray = [], [], {}
    for photo, label, name in parsedCsv:
        dataArray += [photo]
        labelArray += [label]
        translationArray.update({label: name})
                
    return dataArray, labelArray, translationArray


def getNameTranslation(folderPath):
    translationArray = {}
    for _, dirs, _ in os.walk(folderPath):
        for dir in dirs:
            [number, name] = dir.split('.',1)
            translationArray.update({int(number): name})
    return translationArray


'''
 Writes csv file with the input "data"
'''
def writeCSV(filename, data, delimiter=','):
    with open(filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=delimiter, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            str = ''
            spamwriter.writerow(str)


# file= '/home/daniel/Desktop/Pics/data.csv'
# file= '/home/daniel/Desktop/Pics/writerTest.csv'
# writeCSV(file, ['hola', 'como', 'estas'])