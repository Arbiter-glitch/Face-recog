import os
import requests
#def  add():

# i="1"  #Add the folder name to this variable
# parent_dir = "C:/Users/apple/Desktop/S8/Project2/facerecog/root/dataset/"

# directory = i #the folder name for the training images transferred through http request ,note that this is the name that will be outputed
# path = os.path.join(parent_dir, directory)
# os.mkdir(path)
# f = open(path, "wb")
# f.write(r.content) # r is the server response image files .content is to write a it as bytes to the  folder 
# f.close()

os.system("python extract_embeddings.py --dataset dataset \
	      --embeddings output/embeddings.pickle \
	      --detector face_detection_model \
	      --embedding-model openface_nn4.small2.v1.t7")
os.system("python train_model.py --embeddings output/embeddings.pickle \
	      --recognizer output/recognizer.pickle \
	      --le output/le.pickle")

#def recognize():
os.system("python recognize.py --detector face_detection_model \
	   --embedding-model openface_nn4.small2.v1.t7 \
	   --recognizer output/recognizer.pickle \
	   --le output/le.pickle \
	   --image images/arnold.jpg") #arnold is the name of the image to be recognized it can be single file that is repeatedly overwritten
