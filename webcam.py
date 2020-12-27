import cv2
cam = cv2.VideoCapture(0)
cam.set(3,480) # ubah lebar cam
cam.set(4,480) # ubah tinggi cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu,1.3,5) # fram scale
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
    cv2.imshow('webcamku',frame)
    # cv2.imshow('webcamku2', abuAbu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()