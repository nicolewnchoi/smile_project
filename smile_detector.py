#Updated as of 11/27: Finally got it to work for 2.5 hours, ugh

import cv2
from PIL import Image, GifImagePlugin

# im = Image.open('sparkling_filter.gif')

#Face and smile classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#haarcascade_frontalface_default.xml can detect any random object
#haarcascade is made by Viola: an old, simple algorithm for detecting one object
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml') #doesn't work for regular smiles

#Grab Webcam feed
webcam = cv2.VideoCapture(0) #'Why_so_serious.mp4
# sparkling_filter = Image.open('sparkling_filter.gif')
sparkling_filter = cv2.imread('sparkling_filter.png')
# sparkling_filter = cv2.VideoCapture('sparkling_filter.gif')


#Show current frame (loops forever/camera runs in real time until one presses the key)
while True:

    #Reads current frame from the webcame video stream
    successful_frame_read, frame = webcam.read()    #read() function reads real time webcamera footage (single frame)
    
    #If there's an error, abort (safechecking)
    if not successful_frame_read:
        break

    #Change to grayscale (able to see faces in black and white by converting the color to black and white)
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(frame_grayscale)

    #Run face detection within each of those faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 200, 50), 4)

        #From 0 to the end, use slice
        #Get the sub frame (using numpy N-dimensional array slicing)
        the_face = frame[y:y+h, x:x+w]

        #Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        #run smiles on the face
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 20) 

        # for (x_, y_, w_, h_) in smiles:
        #     cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 4)

        if len(smiles) > 0: #smiles is the list of smiles 

            cv2.imshow('Sparkling Filter', sparkling_filter)
            # print(sparkling_filter.is_animated)
            # print(sparkling_filter.n_frames)
            #frame[y:y+h, x:x+w] = apply_mask(frame[y:y+h, x:x+w], sparkling_filter)

            # cv2.putText(frame, 'smiling', (x, y+h+40), fontScale = 3, 
            # fontFace = cv2.FONT_HERSHEY_PLAIN, color=(255,255,255)) #y coordinate + height

    # Show the current frame
    cv2.imshow('Smile Detector', frame)

    #Will update include it
    #cv2.imshow('Sparkling Filter', sparkling_filter)

    #first one is a boolean: a successful read or not
    #second one is actual frame (actual image to read on)

    cv2.waitKey(1)
    cv2.destroyAllWindows() #closes all windows

    """
    #Detect faces first
    # faces = face_detector.detectMultiScale(frame_grayscale, scaleFactor = 1.3, minNeighbors = 30) 
    #faces is an array of points of faces: first point in array is personal face and second point being another person
    #detectMultiScale because want to detect faces of many sizes

    # smiles = smile_detector.detectMultiScale(frame_grayscale, scaleFactor = 1.7, minNeighbors = 20) 
    #scaleFactor: optimization of blurring (how much want to blur image to detect faces, higher number, more blurred)
    #if blur a lot, help contrasts the objects (determine if there is a face or a cutrain), 
    #minNeighbors: 20 neighboring rectangles to be counted as a smile, 
    #there must be 20 redundant rectangles to group into a smile detection
    #always use the parameters: 1.7 and 20 for smiles

    #Run face detection within each of those faces
    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 200, 50), 4)
        #cv2.rectangle(image, start_point (top left), end_point (bottom right), color, thickness)

        #From 0 to the end, use slice
        #Get the sub frame (using numpy N-dimensional array slicing)
        the_face = frame[y:y+h, x:x+w] #note: this works in numpy, not vanilla Python

        #Change to grayscale
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        #run smiles on the face
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor = 1.7, minNeighbors = 10) 

        #Find all smiles in the face
        #within the face, need to find the smiles
        for(x_, y_, w_, h_) in smiles:

            #draw all the rectangles around the face
            cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (100, 200, 50), 4)

        if len(smiles) > 0: #smiles is the list of smiles 
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale = 3, 
            fontFace = cv2.FONT_HERSHEY_PLAIN, color=(255,255,255)) #y coordinate + height

    # Show the current frame
    cv2.imshow('Smile Detector', frame)
    #first one is a boolean: a successful read or not
    #second one is actual frame (actual image to read on)

    #Display
    if cv2.waitKey(1) & 0xff == ord('q'):  #runs the screen every 1 millisecond
        break

#Cleanup
webcam.release()
cv2.destroyAllWindows() #closes all windows
print("Code Completed")

#Code ran without errors
# read("Code ran without errors")



#Show current frame
while True: 
    cv2.imshow('Why So Serious?', frame)

webcam.release()
cv2.destoryAllWindows()

print ("Whats up")
"""
