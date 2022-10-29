from turtle import width
import numpy as np
import cv2

# 读取图片并缩放方便显示
img = cv2.imread("C:\\Users\\17205\\Desktop\\1.png")
height, width = img.shape[:2]
gt = np.zeros((height,width))
drawing =False
x1,y1,x2,y2=-1,-1,-1,-1
pre_img = img.copy()
def cal_value(x1,y1,x2,y2,gt, output)->str: 
    r1 = output[x1:x2,y1:y2].sum()
    r2 = gt[x1:x2,y1:y2].sum()
    return f"sum(gt):{r1},"+f"sum(output):{r2},"+f"MAE{r1-r2}"


def dc(event,x,y,flags,param):
    global x1,y1,x2,y2,drawing,img,gt,output
    if event==cv2.EVENT_LBUTTONDOWN:

        img = pre_img.copy()
        drawing=True
        x1,y1=x,y
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            img = pre_img.copy()
            cv2.rectangle(img,(x1,y1),(x,y),(0,255,0),2)

    elif event==cv2.EVENT_LBUTTONUP:
        
        x2,y2 = x,y
        # begin to calculate
        text = cal_value(x1,y1,x2,y2,gt,pre_img)
        cv2.putText(img, text, (40, 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
        print(text)
        drawing=False

while(1):
    cv2.imshow('img',img)
    cv2.setMouseCallback('img',dc)
    k=cv2.waitKey(1)&0xFF
    if k==27:
        break
