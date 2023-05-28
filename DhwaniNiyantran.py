import mediapipe as mp
import cv2
import time
import math
import numpy as np



#setting camera
cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,460)
cap.set(10,100)

#setting time for fps 
cTime=0
pTime=0

x1,y1,x2,y2=0,0,0,0
midx,midy=0,0
length=0
vol=0
volBar=400


from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()

#to identify the volume rage 
voulumeRange=volume.GetVolumeRange()

minVol=voulumeRange[0]
maxVol=voulumeRange[1]

#accordingly 0==max volume,and minvolume is -96
volume.SetMasterVolumeLevel(0.0, None)











#setting MediaPipe 
mpHands=mp.solutions.hands
hands=mpHands.Hands(max_num_hands=1, min_detection_confidence=0.6)
mpDraw=mp.solutions.drawing_utils


while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,handLms)
            for id,lm in enumerate(handLms.landmark):
                #print(id,lm.x,lm.y)
                
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                #print(id,cx,cy)
                if id==4:
                    cv2.circle(img,(cx,cy),8,(255,255,0))
                    x1,y1=cx,cy

                if id==8:
                    cv2.circle(img,(cx,cy),8,(255,255,0))
                    x2,y2=cx,cy

                cv2.line(img,(x1,y1),(x2,y2),(255,255,0),3)
                midx,midy=int((x1+x2)/2),int((y1+y2)/2)
                cv2.circle(img,(midx,midy),10,(255,0,0),cv2.FILLED)
                length=math.hypot(x2-x1,y2-y1)
                print(length)
                if length<50:
                    cv2.circle(img,(midx,midy),10,(0,255,0),cv2.FILLED)

                
                #hand range is 50 to 200
                #volume range is -95 to 0
                # np.interp is a function to change the given interval into required interval
                vol=np.interp(length,[50,200],[minVol,maxVol])
                volBar=np.interp(length,[50,200],[400,150])
                print(vol)
                volume.SetMasterVolumeLevel(vol, None)
                '''if length < 100:
                   print('line is less than 100')
                else:
                   print('line is greater thann 100')   ''' 

                

                

    cv2.rectangle(img,(50,150),(80,400),(0,255,0),3)  
    cv2.rectangle(img,(50,int(volBar)),(80,400),(0,255,0),cv2.FILLED)  


    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,'fps'+str(int(fps)),(20,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
     

    cv2.imshow('Image',img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break