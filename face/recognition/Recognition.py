'''
Created on Sep 23, 2016

@author: daniel
'''

import numpy as np
import cv2
import sys

from main import preprocessedFaces



'''
Start training from the collected faces.
The face recognition algorithm can be one of these and perhaps more, depending on your version of OpenCV, which must be atleast v2.4.1:
   "FaceRecognizer.Eigenfaces":  Eigenfaces, also referred to as PCA (Turk and Pentland, 1991).
   "FaceRecognizer.Fisherfaces": Fisherfaces, also referred to as LDA (Belhumeur et al, 1997).
   "FaceRecognizer.LBPH":        Local Binary Pattern Histograms (Ahonen et al, 2006).

'''
def learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm):
    print "Learning the collected faces using the {0} algorithm...".format(facerecAlgorithm)
    
    # Make sure the "contrib" module is dynamically loaded at runtime
    # Requires OpenCV v2.4.1 or later (from June 2012), otherwise the FaceRecognizer will not compile or run
#     haveContribModule =  
    
#     if not haveContribModule:
#         print "contrib module is needed for facerecognizer"
#         sys.exit()

    # create recognizer depending on the defined algorithm
    if facerecAlgorithm == 'Fisherfaces':
        recognizer = cv2.createFisherFaceRecognizer()
    elif facerecAlgorithm == 'Eigenfaces':
        recognizer = cv2.createEigenFaceRecognizer()
    else:
        recognizer = cv2.createLBPHFaceRecognizer()
    
    recognizer.train(preprocessedFaces, faceLabels)
    
    return recognizer
    
'''
 Show the internal face recognition data to help debugging
'''
def showTrainingDebugData(model, faceWidth, faceHeight):
    
    averageFaceRow = model.mean()
    print ""
    

'''
 Generate an approximately reconstructed face by back-projecting the eigenvectors and eigenvalues 
 of the given (preprocessed) face
'''
def reconstructFace(model, preprocessedFace):
    # Since we can only reconsctruct the face for some types of FaceRecognizer models (Eigenfaces or Fisherfaces),
    # We should surround the OpenCV calls by try/except block so we don'r crash for other models.
    try:
        
        # Get some required data from the FaceRecognizer model
        eigenvectors = model.get("eigenvectors")
        averageFaceRow = model.get("mean")
        
        faceHeight = len(preprocessedFaces)
        reshapedPreprocessedFace = cv2.cv.ReshapeMatND(preprocessedFace, 1, 1)
        
        # Project the input image onto the PCA subspace
        projection = subspaceProject(eigenvectors, averageFaceRow, reshapedPreprocessedFace)
        # printMatInfo = (projection, 'projection')
        
        #Generate the reconstructed face back from the PCA subspace
        reconstructionRow = subspaceReconstruct(eigenvectors, averageFaceRow, projection, len(preprocessedFace[0]), len(preprocessedFace))
        # printMatInfo(reconsutructionRow, 'reconstructionRow')
        
        # Convert the float row matrix to a regular 8 bit image. Note that we
        # should't use "getImageFrom1DFloatMat()" because we don't want to normalize 
        # the data since it is already at the perfect scale
        
        # Make it a rectangular shaped image instead of a single row
        reconstructionMat = cv2.cv.Reshape(reconstructionRow, 1, faceHeight)
        # Convert the floating-point pixels to regular 8-bit unchar pixels
        reconstructedFace = np.uint8(reconstructionMat)
        
        return reconstructedFace
        
    except Exception as e:
        print e.__str__()
        return np.array()
    
    
'''
 Compare two images by getting the L2 error (square-root of sum of squared error).
'''
def getSimilarity(a,b):
    rowsA = len(a)
    rowsB = len(b)
    
    if rowsA > 0:
        colsA = len(a[0])
        colsB = len(b[0])
        
    if rowsA > 0 and rowsA == rowsB and colsA > 0 and colsA == colsB:
        # Calculate difference
        errorL2 = cv2.norm(a,b,cv2.cv.CV_L2)
        
        #Convert to a reasonable scale, since L2 error is summed across all pixels of the image
        similarity = errorL2 / float(rowsA * colsA) 
        return similarity
    else: 
        return 100000000.0
    


'''
 Got these from Scott Lobdell in stackoverflow 
 @link http://stackoverflow.com/questions/19756176/opencv-facerecognition-subspaceproject-and-subspacereconstruct-methods-in-pytho
 
'''
# projects samples into the LDA subspace
def subspaceProject(eigenvectors_column, mean, source):
    source_rows = len(source)
    source_cols = len(source[0])

    if len(eigenvectors_column) != source_cols * source_rows:
        raise Exception("wrong shape")

    flattened_source = []
    for row in source:
        flattened_source += [float(num) for num in row]
    flattened_source = np.asarray(flattened_source)
    delta_from_mean = cv2.subtract(flattened_source, mean)
    # flatten the matrix then convert to 1 row by many columns
    delta_from_mean = np.asarray([np.hstack(delta_from_mean)])

    empty_mat = np.array(eigenvectors_column, copy=True)  # this is required for the function call but unused
    result = cv2.gemm(delta_from_mean, eigenvectors_column, 1.0, empty_mat, 0.0)
    return result


# reconstructs projections from the LDA subspace
def subspaceReconstruct(eigenvectors_column, mean, projection, image_width, image_height):
    if len(eigenvectors_column[0]) != len(projection[0]):
        raise Exception("wrong shape")

    empty_mat = np.array(eigenvectors_column, copy=True)  # this is required for the function call but unused
    # GEMM_2_T transposes the eigenvector
    result = cv2.gemm(projection, eigenvectors_column, 1.0, empty_mat, 0.0, flags=cv2.GEMM_2_T)

    flattened_array = result[0]
    flattened_image = np.hstack(cv2.add(flattened_array, mean))
    flattened_image = np.asarray([np.uint8(num) for num in flattened_image])
    all_rows = []
    for row_index in xrange(image_height):
        row = flattened_image[row_index * image_width: (row_index + 1) * image_width]
        all_rows.append(row)
    image_matrix = np.asarray(all_rows)
    image = normalizeHist(image_matrix)
    return image

def normalizeHist(face):
    face_as_mat = np.asarray(face)
    equalized_face = cv2.equalizeHist(face_as_mat)
    equalized_face = cv2.cv.fromarray(equalized_face)                                                                                                                                                                                 
    return equalized_face
   