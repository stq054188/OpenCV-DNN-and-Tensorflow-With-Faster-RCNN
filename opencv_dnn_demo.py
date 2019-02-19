import cv2 as cv
import os

imgList = []
#dir = "E:/Practice/TensorFlow/DataSet/fingermark/VOC2012/JPEGImages/"
dir = "E:/Practice/TensorFlow/DataSet/danger/VOC2012/JPEGImages/"
#dir = "D:/hand_data/VOC2012/JPEGImages/"
for root, dirs, files in os.walk(dir):
    for file in files:
        str1 = os.path.join(root,file)
        if str1.find(".jpg")!= -1:
            imgList.append(str1)
#print(imgList)

cvNet = cv.dnn.readNetFromTensorflow('./model_rcnn3/frozen_inference_graph.pb',
                                   './model_rcnn3/graph.pbtxt')
#cvNet = cv.dnn.readNetFromTensorflow('D:/tensorflow/handset/export/frozen_inference_graph.pb',
#                                   'D:/tensorflow/handset/export/graph.pbtxt')

num = 0 
for i in range(0,len(imgList)):
    num = num + 1
    #img = cv.imread('E:/Practice/TensorFlow/DataSet/fingermark/VOC2012/JPEGImages/15.jpg')
    img = cv.imread(imgList[i])
    rows = img.shape[0]
    cols = img.shape[1]

    start = cv.getTickCount()
    cvNet.setInput(cv.dnn.blobFromImage(img, size=(600, 600), swapRB=True, crop=False))
    cvOut = cvNet.forward()

    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        if score > 0.3:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
            cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), thickness=4)

    end = cv.getTickCount()
    use_time = (end - start) / cv.getTickFrequency()
    print("use_time:%0.3fs" % use_time)

    if num < 10:
        winName = "img%d"%num
        cv.imshow(winName, img)
    else:
        break
cv.waitKey(0)
cv.destroyAllWindows()
