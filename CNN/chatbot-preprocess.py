import os
import cv2

path_1 = "image/zz" #collected image class 1
path_2 = "image/zb" #collected image class 2
path_3 = "image/ts" #collected image class 3
path_t="data" #save preprocessed images

id=0
IMG_SIZE = 32
# for (path, dirs, files) in os.walk(path_1):
#     for filename in files:
#         newname = "0_{}.jpg".format(id)
#         id+=1
#         os.rename(path_1 + "\\" + filename, path_t + "\\" + newname)
#
# for (path, dirs, files) in os.walk(path_2):
#     for filename in files:
#         newname = "1_{}.jpg".format(id)
#         id+=1
#         os.rename(path_2 + "\\" + filename, path_t + "\\" + newname)
#
# for (path, dirs, files) in os.walk(path_3):
#     for filename in files:
#         newname = "2_{}.jpg".format(id)
#         id+=1
#         os.rename(path_3 + "\\" + filename, path_t + "\\" + newname)

#
# for filename in os.listdir(path_t):
#     print(filename)
#     path= "data"+ "\\" +filename
#     img = cv2.imread(path)
#
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     cv2.imwrite("data_m"+ "\\"+filename,img)

# for filename in os.listdir("test"):
#     print(filename)
#     path= "test"+ "\\" +filename
#     img = cv2.imread(path)
#
#     img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#     cv2.imwrite("test_s"+ "\\"+filename,img)
