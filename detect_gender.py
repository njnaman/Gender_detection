
# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import time
import cvlib as cv



dwnld_link = r"D:\NEC trial1\gender-detection-keras\pre-trained\gender_detection.model"
model_path = get_file("gender_detection.model", dwnld_link,
                     cache_subdir="pre-trained", cache_dir=os.getcwd())

images =[]
outputImages=[]
mins=0
video_capture = cv2.VideoCapture(0)
# Check success

while(mins!=3):
    if not video_capture.isOpened():
        raise Exception("Could not open video device")
    # Read picture. ret === True on success
    ret, image = video_capture.read()
    images.append(image)
    time.sleep(0.6)
    mins=mins+1
    # Close device
video_capture.release()

# read input image
imageTackled=0
malecount=0
femalecount=0

while(imageTackled!=3):
    image =images[imageTackled]
    if image is None:
        print("Could not read input image")
        exit()

    # load pre-trained model
    model = load_model(model_path)

    # detect faces in the image
    face, confidence = cv.detect_face(image)

    classes = ['man','woman']
    #male,female=0,0
    # loop through detected faces
    for idx, f in enumerate(face):

         # get corner points of face rectangle       
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(image[startY:endY,startX:endX])

        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0]
        #print(conf)
        #print(classes)


        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]

        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10

        typ = label.split(":")
        #print(f" -------------- {typ[0]}")
        if(typ[0]=="man"):
            malecount=malecount+1
        else:
            femalecount=femalecount+1

        # write label and confidence above face rectangle
        cv2.putText(image, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

        #print(f" male : --- {male} \nfemale : --- {female}")
        outputImages.append(image)
        imageTackled=imageTackled+1

malecount=malecount/3
femalecount=femalecount/3
if(malecount>femalecount):
    print(f"{malecount} ** {femalecount} ************************  aadmi hoo m")
else:
    print(f"{malecount} ** {femalecount} ************************  Aurat hoo m")

# display output
cv2.imshow("gender detection", outputImages[2])

# press any key to close window           
cv2.waitKey(0)

# save output
cv2.imwrite("gender_detection1.jpg", outputImages[0])
cv2.imwrite("gender_detection2.jpg", outputImages[1])
cv2.imwrite("gender_detection3.jpg", outputImages[2])
# release resources
cv2.destroyAllWindows()
