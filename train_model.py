#import necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse 
import pickle

#construct the argumet parser and parse the arguments
ap=argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True,help="path to serialized db of facial embeddings")
ap.add_argument("-r", "--recognizer", required=True,help="path to output model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,help="path to output label encoder")
args=vars(ap.parse_args())

#load face embeddings
print("[Info] loading face embeddings ")
data=pickle.loads(open(args["embeddings"],"rb").read())

#encode the labels
print("[Info] encoding labels...")
le=LabelEncoder()
labels=le.fit_transform(data["names"])

#train the model used to accept the 128-d face embeddings andthen produce actual face recognition
print("[Info] training model...")
recognizer=SVC(C=1.0,kernel="linear",gamma=10.0,random_state=0,probability=True)
recognizer.fit(data["embeddings"],labels)

#write actual face recognition model to disk
f=open(args["recognizer"],"wb")
f.write(pickle.dumps(recognizer))
f.close()

#write label encoder to disk
f=open(args["le"],"wb")
f.write(pickle.dumps(le))
f.close()