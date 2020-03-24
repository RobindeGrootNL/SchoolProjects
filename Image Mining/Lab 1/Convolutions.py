import numpy as np
import cv2

from matplotlib import pyplot as plt

#Read grayscale image and conversion to float64
img=np.float64(cv2.imread('./Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Image dimension:",h,"rows x",w,"columns")

#Direct method
t1 = cv2.getTickCount()
img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(1,w-1):
    val = 5*img[y, x] - img[y-1, x] - img[y, x-1] - img[y+1, x] - img[y, x+1] 
    img2[y,x] = min(max(val,0),255)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Direct method:",time,"s")


#Direct method gradient computation
gradient = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)

for x in range(0,h-1):
    for y in range(0,w-1):
        gradient_x = (img[x+1, y] - img[x, y])/2
        gradient_y = (img[x, y+1] - img[x, y])/2
        gradient[x,y] = np.sqrt(gradient_x**2 + gradient_y**2)
        


plt.subplot(121)
plt.imshow(img2,cmap = 'gray')
plt.title('Convolution - Direct method')

#Method filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
img3 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Method filter2D :",time,"s")

plt.subplot(122)
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Convolution - filter2D')

plt.show()
