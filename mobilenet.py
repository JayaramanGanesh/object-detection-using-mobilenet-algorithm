import numpy as np
import imutils
import cv2
import time

prototxt = "MobileNetSSD_deploy.prototxt"
model = "MobileNetSSD_deploy.caffemodel"
confThresh = 0.2

classes = ["background","bicycle","bird","boat","bottle","car","cat","chair","person","tvmonitor","train","bus","mobile",
            "aeroplane","cow","diningtable","dog","horse","motorbike","pottedplant","sheep","sofa"]
colors = np.random.uniform(0, 255, size = (len(classes),3))
print("model loadding.......")
net = cv2.dnn.readNetFromCaffe(prototxt,model)
print("model loadding completed")


vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _,frame = vs.read()
    frame = imutils.resize(frame, width=500)
    (h,w) = frame.shape[:2]
    imresize = cv2.resize(frame,(300, 300))
    blob = cv2.dnn.blobFromImage(imresize,0.007843,(300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    detshape = detections.shape[2]
    for i in np.arange(0,detshape):
        confidence = detections[0,0,i,2]
        if confidence > confThresh:
            idx = int(detections[0,0,i,1])
            print("class id : ",detections[0,0,i,1])
            box = detections[0,0,i,1] * np.array([w, h, w, h])
            print("box coord :",detections[0,0,i,3:7])
            (startX, startY, endX, endY) = box.astype("int")
            label = "{} : {:.2f} %".format(classes[idx],confidence * 100)
            cv2.rectangle(frame,(startX,startY),(endX,endY),colors[idx],2)
            if startY - 15 > 15:
                y = startY - 15
            else:
                y = startY + 15
            cv2.putText(frame,label,(startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx],2)

    cv2.imshow("frame",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break



vs.release()
vs.destroyAllWindows()        

