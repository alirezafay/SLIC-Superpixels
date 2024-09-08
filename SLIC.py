#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[2]:


image1 = cv2.imread('brain.JPG')
gray_img1 = cv2.imread('brain.JPG',cv2.IMREAD_GRAYSCALE)


# In[3]:


image = np.zeros((2048,2048,3),dtype=np.uint8)
gray_img = np.zeros((2049,2049),dtype=np.uint8)


# In[4]:


image[0:2040,0:2040,:]= image1
gray_img[0:2040,0:2040]= gray_img1


# In[5]:


kernel_x = np.array([[-1,-2,-1], [0, 0, 0],[1,2,1]])
difference_x = cv2.filter2D(gray_img, -1, kernel_x)
kernel_y = np.array([[-1,0,-1], [-2, 0, 2],[-1,0,-1]])
difference_y = cv2.filter2D(gray_img, -1, kernel_y)
grad_mag = np.sqrt(np.power(difference_y,2) + np.power(difference_x,2))


# In[6]:


n_centr = 64
rows = 8
cols = 8
center_list = [[[] for j in range(cols)] for i in range(rows)]
updt_center = [[[] for j in range(cols)] for i in range(rows)]
neigh = np.zeros((5,5))
n = 8
for i in range(n):
    for j in range(n):
        center_list[i][j] = ([128+i*256,128+j*256])
for i in range(n):
    for j in range(n):       
        for m in range(5):
            for z in range(5):
                neigh[m,z] = grad_mag[center_list[i][j][0]-2+m,center_list[i][j][1]-2+z]
        min_index = np.argmin(neigh)
        row_index = min_index // neigh.shape[1]
        col_index = min_index % neigh.shape[1]
        updt_center[i][j] = ([center_list[i][j][0]-2+row_index,center_list[i][j][1]-2+col_index])


# In[7]:


alpha = 0.4
image_seg = np.zeros(gray_img.shape)
for i in range(gray_img.shape[0]-1):
    for j in range(gray_img.shape[0]-1):
        cost_f = np.zeros((8,8))
        d_bright = np.zeros((8,8))
        d_elud = np.zeros((8,8))
        for k in range(8):
            for g in range(8):
                if np.sqrt(np.power(i-updt_center[k][g][0],2) + np.power(j-updt_center[k][g][1],2)) < 200:           
                    d_bright[k,g] = np.sqrt(np.power(image[i,j,0] - image[updt_center[k][g][0],updt_center[k][g][1],0],2) + np.power(image[i,j,1] - image[updt_center[k][g][0],updt_center[k][g][1],1],2) + np.power(image[i,j,2] - image[updt_center[k][g][0],updt_center[k][g][1],2],2))
                    d_elud[k,g] = np.sqrt(np.power(i-updt_center[k][g][0],2) + np.power(j-updt_center[k][g][1],2))
                    cost_f[k,g] = d_bright[k,g] + alpha*d_elud[k,g]
                else:
                    d_bright[k,g] = 100000
                    d_elud[k,g] = 100000
                    cost_f[k,g] = 200000
        min_cost_i = np.argmin(cost_f)
        row_cost_i = min_cost_i // cost_f.shape[1]
        col_cost_i = min_cost_i % cost_f.shape[1]
        image_seg[i,j] = col_cost_i + 8*row_cost_i + 1


# In[118]:


image_seg_final = image_seg[0:2047,0:2047]
gradient_x_seg = np.zeros(image_seg_final.shape)
gradient_y_seg = np.zeros(image_seg_final.shape)
for i in range(1,len(image_seg_final)):
    gradient_x_seg[i,:] =  image_seg_final[i,:] - image_seg_final[i-1,:] 
    gradient_y_seg[:,i] =  image_seg_final[:,i] - image_seg_final[:,i-1]
gradient_mag_b = np.sqrt(np.square(gradient_x_seg) + np.square(gradient_y_seg*8))
ret,difference_seg_th = cv2.threshold(gradient_mag_b, 1, 255, cv2.THRESH_BINARY)    


# In[119]:


mask = np.zeros((2048,2048,3))
mask[0:2047,0:2047,0] = difference_seg_th
mask[0:2047,0:2047,1] = difference_seg_th
mask[0:2047,0:2047,2] = difference_seg_th
mask = np.uint8(mask)


# In[129]:


output = cv2.addWeighted(image, 1, mask, 1, 0)
output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
plt.imshow(output)


# In[121]:


n_centr2 = 256
rows2 = 16
cols2 = 16
center_list2 = [[[] for j in range(cols2)] for i in range(rows2)]
updt_center2 = [[[] for j in range(cols2)] for i in range(rows2)]
neigh2 = np.zeros((5,5))
n2 = 16
for i in range(n2):
    for j in range(n2):
        center_list2[i][j] = ([64+i*128,64+j*128])
for i in range(n2):
    for j in range(n2):       
        for m in range(5):
            for z in range(5):
                neigh[m,z] = grad_mag[center_list2[i][j][0]-2+m,center_list2[i][j][1]-2+z]
        min_index2 = np.argmin(neigh2)
        row_index2 = min_index2 // neigh2.shape[1]
        col_index2 = min_index2 % neigh2.shape[1]
        updt_center2[i][j] = ([center_list2[i][j][0]-2+row_index2,center_list2[i][j][1]-2+col_index2])


# In[131]:


alpha2 = 0.65
image_seg2 = np.zeros(gray_img.shape)
for i in range(gray_img.shape[0]-1):
    for j in range(gray_img.shape[0]-1):
        cost_f2 = np.zeros((16,16))
        d_bright2 = np.zeros((16,16))
        d_elud2 = np.zeros((16,16))
        for k in range(16):
            for g in range(16):
                if np.sqrt(np.power(i-updt_center2[k][g][0],2) + np.power(j-updt_center2[k][g][1],2)) < 100:           
                    d_bright2[k,g] = np.sqrt(np.power(image[i,j,0] - image[updt_center2[k][g][0],updt_center2[k][g][1],0],2) + np.power(image[i,j,1] - image[updt_center2[k][g][0],updt_center2[k][g][1],1],2) + np.power(image[i,j,2] - image[updt_center2[k][g][0],updt_center2[k][g][1],2],2))
                    d_elud2[k,g] = np.sqrt(np.power(i-updt_center2[k][g][0],2) + np.power(j-updt_center2[k][g][1],2))
                    cost_f2[k,g] = d_bright2[k,g] + alpha2*d_elud2[k,g]
                else:
                    d_bright2[k,g] = 100000
                    d_elud2[k,g] = 100000
                    cost_f2[k,g] = 200000
        min_cost_i2 = np.argmin(cost_f2)
        row_cost_i2 = min_cost_i2 // cost_f2.shape[1]
        col_cost_i2 = min_cost_i2 % cost_f2.shape[1]
        image_seg2[i,j] = col_cost_i2 + 16*row_cost_i2 + 1


# In[132]:


image_seg_final2 = image_seg2[0:2047,0:2047]
gradient_x_seg2 = np.zeros(image_seg_final2.shape)
gradient_y_seg2 = np.zeros(image_seg_final2.shape)
for i in range(1,len(image_seg_final2)):
    gradient_x_seg2[i,:] =  image_seg_final2[i,:] - image_seg_final2[i-1,:] 
    gradient_y_seg2[:,i] =  image_seg_final2[:,i] - image_seg_final2[:,i-1]
gradient_mag_b2 = np.sqrt(np.square(gradient_x_seg2) + np.square(gradient_y_seg2*8))
ret2,difference_seg_th2 = cv2.threshold(gradient_mag_b2, 1, 255, cv2.THRESH_BINARY)    


# In[133]:


mask2 = np.zeros((2048,2048,3))
mask2[0:2047,0:2047,0] = difference_seg_th2
mask2[0:2047,0:2047,1] = difference_seg_th2
mask2[0:2047,0:2047,2] = difference_seg_th2
mask2 = np.uint8(mask2)


# In[135]:


output2 = cv2.addWeighted(image, 1, mask2, 1, 0)
output2 = cv2.cvtColor(output2, cv2.COLOR_BGR2RGB)
plt.imshow(output2)


# In[ ]:




