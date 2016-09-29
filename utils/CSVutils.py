'''
Created on Sep 29, 2016

@author: daniel
'''

import csv
import subprocess


'''
 Update the photo list csv with labels
'''
def updatePhotoList(path, filename='test.sh', flag=True):
    if flag:
        subprocess.call('' + path + filename)
    

'''
 Read a CSV file and return 
 delimiter defines the separation character between columns in the CSV file
 type will decide if we are reading:
    0 csv file with picture filenames
    1 preprocessed faces or pretrained data files
 Returns an array of tuples that contains the data and its label
'''
def readCSV(filename, delimiter=',', type=0):
    out = []
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
        for row in spamreader:
            if type is 1:
                # Return an array with the details and the label separately
                out+= [(row[:len(row)-1], row[len(row)-1])]
            else:
                # For reading csv with photo names
                out += [(row[0], int(row[1]))]
    return out

'''
 Gets the array from the readCSV function and returns two different arrays:
   dataArray - contains the data of the file, preprocessed face or pretrained data 
   labelArray - contains the name of the person to which this data belongs to.
'''
def getPhotoAndLabel(filename, type=None):
#     updatePhotoList('/home/daniel/Desktop/Pics/', 'test.sh', True)
    parsedCsv = readCSV(filename)
    dataArray, labelArray = [], [] 
    for photo, label in parsedCsv:
        dataArray += [photo]
        labelArray += [label]
        
    return dataArray, labelArray


# print getPhotoAndLabel(file)[0]
# print getPhotoAndLabel(file)[1]

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