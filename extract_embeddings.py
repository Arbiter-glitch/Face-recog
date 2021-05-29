#import necessary packages
from imutils import paths
import numpy as np
import imutils
import pickle
import argparse
import cv2
import os

# construct the argument parser and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-i","--dataset",required=True,help="path to input directory of faces + images")
ap.add_argument("-e","--embeddings",required=True,help="path to output serialized db of facial embeddings")
ap.add_argument("-d","--detector",required=True,help="path to Opencv's deeplearning face detector")
ap.add_argument("-m","--embedding-model",required=True,help="path to Opencv's deeplearning face embedding model")
ap.add_argument("-c","--confidence", type=float, default=0.55,help="minimum probability to filter weak decisions")
args=vars(ap.parse_args())

#load our serialised face detector from disk
print("[Info] loading face detector...")
protoPath=os.path.sep.join([args["detector"],"deploy.prototxt"])
modelPath=os.path.sep.join([args["detector"],"res10_300x300_ssd_iter_140000.caffemodel"])
detector=cv2.dnn.readNetFromCaffe(protoPath,modelPath)

#load our serialised face embedding model from disk
print("[Info] loading face recognizer...")
embedder=cv2.dnn.readNetFromTorch(args["embedding_model"])


#grab paths to input images into the dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))
#initialize our list of extracted facial embeddings and corresponding people names
knownEmbeddings=[]
knownNames=[]
#initialize total no. of faces processed
total=0

#loop over image paths
for (i,imagePath) in enumerate(imagePaths):
    
    
    #extract the person name from the image path
    print("[INFO] processing image{}/{}".format(i+1,len(imagePaths)))
    name=imagePath.split(os.path.sep)[-2]
     
    # load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), 
    # and then grab the image dimensions
    image=cv2.imread(imagePath)
    
    image=imutils.resize(image,width=600)
    (h,w)=image.shape[:2]
     
    #construct blob from image
     
    imageBlob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0,177.0,123.0),swapRB=False,crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
    detector.setInput(imageBlob)
    detections=detector.forward()
    #ensure atleast one face is found
    if len(detections) >0:
        #Note...we are making the assumption that each image has only one face so finding bounding box with largest probability
        #for registering a face image should only contain single person
        i=np.argmax(detections[0,0, :, 2]) 
        
        confidence=detections[0,0,i,2]

        # ensure that the detection with the largest probability
        # also means our minimum probability test (thus helping filter out
		# weak detections
    
        if confidence > args["confidence"]:
        #compute the (x,y) coordinates of the bounding box for the object
            box=detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY)= box.astype("int")

            #extract the face ROI and grab ROI dimensions
            face=image[startY:endY, startX:endX]
            (fH,fW) =face.shape[:2]
        
             #ensure that the face width and height are sufficiently large
            if fW<20 or fH<20:
                 continue
            #construct blob of face ROI then pass the blob through 
            #our face embedding model to obtain 128-d quantfication of face
            faceBlob=cv2.dnn.blobFromImage(face,1.0/255,(96,96),(0,0,0),swapRB=True,crop=False)
            embedder.setInput(faceBlob)
            vec=embedder.forward()
        
            #add the name of the person + his face embedding to their respective lists
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total +=1

# dump the facial embeddings + names to disk
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()



