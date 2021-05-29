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
ap.add_argument("-i","--image",required=True,help="path to input images")
ap.add_argument("-d","--detector",required=True,help="path to Opencv's deeplearning face detector")
ap.add_argument("-m","--embedding-model",required=True,help="path to Opencv's deeplearning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,help="path to label encoder")
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

# load the actual face recognition model along with the label encoder
recognizer=pickle.loads(open(args["recognizer"],"rb").read())
le=pickle.loads(open(args["le"],"rb").read())

# load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio), 
# and then grab the image dimensions
image=cv2.imread(args["image"]) 
image=imutils.resize(image,width=600)
(h,w)=image.shape[:2]
     
#construct blob from image
imageBlob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0,177.0,123.0),swapRB=False,crop=False)


# apply OpenCV's deep learning-based face detector to localize
# faces in the input image
detector.setInput(imageBlob)
detections=detector.forward()


#loop over the detections
for i in range (0,detections.shape[2]):
    #extract confidence(i.e.probability) associated with the predictions
    confidence=detections[0,0,i,2]
    #filter out weak detections by ensuring confidence is greater than the min confidence
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
        
        #perform classifications to recognize faces
        preds=recognizer.predict_proba(vec)[0]
        j=np.argmax(preds)
        proba=preds[j]
        name=le.classes_[j]
        
        #draw the bounding box of the face along with the associated probability
        text="{}: {:.2f}%".format(name,proba * 100)
        y= startY-10 if startY-10 > 10 else startY+10
        cv2.rectangle(image, (startX, startY), (endX, endY),(0, 0, 255), 2)
        cv2.putText(image,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)
        print (name)
#show output image
cv2.imshow("Image",image)
cv2.waitKey(0)






