import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

model = YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('Video-76.mp4')
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
print(f"Number of classes: {len(class_list)}")

count = 0
area1 = [(22,12),(13,487),(990,477),(1000,21)]

while True:    
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame,(1020,500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
   
    list=[]    
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        
        if d < len(class_list):
            c = class_list[d]
        else:
            print(f"Warning: Class index {d} is out of range. Max index is {len(class_list)-1}")
            c = "Unknown"
        
        cx = int(x1+x2)//2
        cy = int(y1+y2)//2
        w, h = x2-x1, y2-y1
        
        result = cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
        if result>=0:                           
            cvzone.cornerRect(frame,(x1,y1,w,h),3,2)
            cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
            cvzone.putTextRect(frame,f'cow ghee',(x1,y1),1,1)
            list.append(cx)
    
    cr1=len(list)
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
    cvzone.putTextRect(frame,f'Count: {cr1}',(50,60),1,1)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()