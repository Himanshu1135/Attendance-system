import cv2
import face_recognition as fr
import numpy as np
import os
from datetime import datetime

path = r"D:\Himanshu_data_science\opencv\attendence system\train_img"
mylist = os.listdir(path)                       # mylist is not array list
images = []
names = []
print(mylist)
for i in mylist:
    cimg = cv2.imread(f'{path}/{i}')             # its take path not give mylist
    images.append(cimg)
    names.append(os.path.splitext(i)[0])         # for removing .jpg
# print(images)

def findingfaces(a):                             #for enconding images list that we have given
    encolist = []
    for j in images:
        j = cv2.cvtColor(j,cv2.COLOR_BGR2RGB)    #fr read in rgb
        enco = fr.face_encodings(j)[0]           # [0] for 1 element placement
        encolist.append(enco)
    return (encolist)
encolist = findingfaces(images)
print(len(encolist))
def attendance(name):
    with open(r"D:\Himanshu_data_science\opencv\attendence system\attendence.csv", 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')

cap = cv2.VideoCapture(0)
while True:
    sucess,frames = cap.read()
    frame = cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)

    frameloc = fr.face_locations(frame)
    frameencode = fr.face_encodings(frame,frameloc)   # we have given frameloc for fast encoding

    for fl,fe in zip(frameloc,frameencode):
        match = fr.compare_faces(encolist,fe)
        matchdis = fr.face_distance(encolist,fe)

        matchid = np.argmin(matchdis)                 # give index for minma in list
        if match[matchid]:
            name = names[matchid]
            print(name)
            frame = cv2.putText(frames,name,(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
            attendance(name)


    cv2.imshow("frames",frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break






