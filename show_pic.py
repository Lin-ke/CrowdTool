import cv2
import numpy as np
class picshower():

    def show_pic(self,target,output,img,out_img):
        self.output = output
        self.gt = target
        self.drawing =False
        self.x1,self.y1,self.x2,self.y2=-1,-1,-1,-1
        self.pre_img = img.copy()
        self.pre_out_img = out_img.copy()
        self.img = img
        self.out_img = out_img
        def cal_value()->str: 
            print(self.x1,self.x2,self.y1,self.y2)
            r1 = self.gt[self.y1:self.y2,self.x1:self.x2].sum()
            r2 = self.output[self.y1:self.y2,self.x1:self.x2].sum()
            return f"sum(gt):{r1},"+f"sum(output):{r2},"+f"MAE {abs(r1-r2)}"


        def dc(event,x,y,flags,param):
            if event==cv2.EVENT_LBUTTONDOWN:

                self.img = self.pre_img.copy()
                self.out_img = self.pre_out_img.copy()
                self.drawing=True
                self.x1,self.y1=x,y
            elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
                if self.drawing == True:
                    self.img = self.pre_img.copy()
                    self.out_img = self.pre_out_img.copy()
                    cv2.rectangle(self.img,(self.x1,self.y1),(x,y),(0,255,0),2)
                    cv2.rectangle(self.out_img,(self.x1,self.y1),(x,y),(0,255,0),2)

            elif event==cv2.EVENT_LBUTTONUP:
                
                self.x2,self.y2 = x,y
                # begin to calculate
                text = cal_value()
                cv2.putText(self.out_img, text, (40, 50), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2)
                print(text)
                self.drawing=False
        cv2.namedWindow('img',0)
        cv2.resizeWindow('img',640,480)
        cv2.namedWindow('out_img',0)
        cv2.resizeWindow('out_img',640,480)
        while(1):
            cv2.imshow('img',self.img)
            
            cv2.imshow("out_img",self.out_img)
            cv2.setMouseCallback('img',dc)
            cv2.setMouseCallback('out_img',dc)
            k=cv2.waitKey(10)
            if k==27:
                break
if __name__ == "__main__":
    s = picshower()
    s.show_pic(np.zeros((1000,1000)),np.ones((1000,1000)),cv2.imread("C:\\Users\\17205\Desktop\\1.png"),cv2.imread("C:\\Users\\17205\Desktop\\1.png"))