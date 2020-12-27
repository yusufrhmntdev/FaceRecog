import cv2,os
wajahDir = 'dataWajah'
cam = cv2.VideoCapture(0)
cam.set(3,480) # ubah lebar cam
cam.set(4,480) # ubah tinggi cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml')
faceID = input("masukan Face Id [tekan enter]  : ")
print("tatap wajah ke dalam webcam.tunggu proses pengambilan data wajah selesai ... ")
ambilData = 0

while True:
    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(abuAbu,1.3,5) # fram scale
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        namaFile = 'muka.'+str(faceID)+'.'+str(ambilData)+'.jpg'
        cv2.imwrite(wajahDir+'/'+namaFile,frame)
        ambilData += 1
        roiAbuAbu = abuAbu[y:y+h,x:x+w]
        roiWarna = frame[y:y+h,x:x+w]
        eyes = eyeDetector.detectMultiScale(roiAbuAbu)
        for(xe,ye,we,he) in eyes:
            cv2.rectangle(roiWarna,(xe,ye),(xe+we,ye+he),(0,0,255),1)
    cv2.imshow('webcamku',frame)
    # cv2.imshow('webcamku2', abuAbu)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
    elif ambilData > 10:
        break
print("kelar")
cam.release()
cv2.destroyAllWindows()