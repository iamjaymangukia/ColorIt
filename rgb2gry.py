import cv2
import glob
import os

# os.mkdir('D:/sdp_6th/color/GrayScale_Images')
# images_path = glob.glob('D:/ColorIt/landscape Images/color/*.jpg')
images_path = 'D:/ColorIt/landscape Images/color'
i = 0
# for image in images_path:
#     print (image)
    # img = cv2.imread(image)
    # gray_images = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # cv2.imshow('Gray Images', gray_images)
    # cv2.imwrite('D:/ColorIt/landscape Images/gray/img%i.jpg' %i, gray_images)
    # i += 1 
    # cv2.waitKey(600)
    # cv2.destroyAllWindows()

for image in os.listdir(images_path):
    print (image)
    path = images_path + '/' + image
    img = cv2.imread(path)
    gray_images = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    path = 'D:/ColorIt/landscape Images/gray/' + image
    cv2.imwrite(path ,gray_images)
    i += 1 
    cv2.waitKey(600)
    cv2.destroyAllWindows()