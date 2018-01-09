#Face Recognition

#Importing the libraries
import cv2

#Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Defining a function that will do the detections
#grey=black and white, frame=
#1.3->scale factor
#5->minimum number of zones or neighbours
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #x=, y=, z=
    #x,y:are coordinates of upper rectangel w: width, h: height of rectangle
    #(255, 0, 0)=color codesBGR code
    #OpenCV isung BGR instead of RGB.
    #2=thikness of rectangle
    for (x, y, w, h) in faces:
        cv2.reactangle(frame, (x,y),(x+w, y+h), (255, 0, 0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0),2)
    return frame

#Doing some Face Recognition with the webcam
#0=internal webcam
#1=external webcam
video_capture = cv2.VideoCapture(0)

#infinite loop
while True:
    _, frame=video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#converting to color to black and white
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas)#show output in animated show
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break


video_capture.release()
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
            