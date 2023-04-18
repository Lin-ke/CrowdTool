import cv2
import numpy as np
import random
class picshower():
    def cal_value(self)->str: 
        print(self.x1,self.x2,self.y1,self.y2)
        r1 = float(self.gt[self.y1:self.y2,self.x1:self.x2].sum())
        r2 = float(self.output[self.y1:self.y2,self.x1:self.x2].sum())
        return f"sum(gt):{r1},"+f"sum(output):{round(r2, 2)},"+f"MAE {round(abs(r1-r2),2)}"
    def show_pic(self,target,output,imgs,save = None):
        self.est = str(round(float(output.sum()),2))
        self.gtcount = str(float(target.sum()))
        self.output = output
        self.gt = target
        self.drawing =False
        self.x1,self.y1,self.x2,self.y2=-1,-1,-1,-1
        self.pre_imgs = []
        cv2.putText(imgs[2],"Est:{}".format(self.est),(460,460),cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255), 2)
        cv2.putText(imgs[3],"Gt:{}".format(self.gtcount),(460,460),cv2.FONT_HERSHEY_PLAIN, 2.0, (255,255,255), 2)
        for img in imgs:
            self.pre_imgs.append(img.copy())
        self.num = len(imgs)
        self.imgs = imgs

        def dc(event,x,y,flags,param):
            if event==cv2.EVENT_LBUTTONDOWN:
                for i in range(self.num):
                    self.imgs[i] = self.pre_imgs[i].copy()
                self.drawing=True
                self.x1,self.y1=x,y
            elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
                if self.drawing == True:
                    for i in range(self.num):
                        self.imgs[i] = self.pre_imgs[i].copy()
                        cv2.rectangle(self.imgs[i],(self.x1,self.y1),(x,y),(0,255,0),2)

            elif event==cv2.EVENT_LBUTTONUP:
                
                self.x2,self.y2 = x,y
                # begin to calculate
                text = self.cal_value()
                cv2.putText(self.imgs[1], text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,255,255), 2)
                print(text)
                self.drawing=False
        for i in range(self.num):
            cv2.namedWindow('img'+str(i),0)
            cv2.resizeWindow('img'+str(i),(self.imgs[i].shape[1],self.imgs[i].shape[0]))
        while(1):
            for i in range(self.num):
                cv2.imshow('img'+str(i),self.imgs[i])
                cv2.setMouseCallback('img'+str(i),dc)
            k=cv2.waitKey(10)
            if k==27:
                break
            if k==ord("s"):
                print("save")
                for i in range(self.num):
                    basename = "./" + str(save)+"_"+str(random.randint(1,100))+".jpg"
                    cv2.imwrite(basename, self.imgs[i])
if __name__ == "__main__":
    s = picshower()
    s.show_pic(np.zeros((1000,1000)),np.ones((1000,1000)),cv2.imread("C:\\Users\\17205\Desktop\\1.png"),cv2.imread("C:\\Users\\17205\Desktop\\1.png"))
