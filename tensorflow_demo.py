import os
import sys
import tarfile
import cv2
import tensorflow as tf
import numpy as np

imgList = []
#dir = "E:/Practice/TensorFlow/DataSet/fingermark/VOC2012/JPEGImages/"
dir = "E:/Practice/TensorFlow/DataSet/danger/VOC2012/JPEGImages/"
for root, dirs, files in os.walk(dir):
    for file in files:
        str1 = os.path.join(root,file)
        if str1.find(".jpg")!= -1:
            imgList.append(str1)
#print(imgList)

from utils import label_map_util
from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_FROZEN_GRAPH = './model_ssd_fpn/frozen_inference_graph.pb'
#PATH_TO_FROZEN_GRAPH = './model_rcnn2/frozen_inference_graph.pb'
PATH_TO_FROZEN_GRAPH = './model_rcnn3/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('./model_ssd_fpn', 'fingermark_label_map.pbtxt')
#PATH_TO_LABELS = os.path.join('./model_rcnn2', 'fingermark_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('./model_rcnn3', 'danger_label_map.pbtxt')

NUM_CLASSES = 2

#start = cv2.getTickCount()
   
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categorys = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
categorys_index = label_map_util.create_category_index(categorys)

##def load_image_into_numpy(image):
##  (im_w, im_h) = image.size
##  return np.array(image.getdata()).reshape((im_h,im_w,3)).astype(np.uint8)
num = 0

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for i in range(0,len(imgList)):
        num = num + 1
        #image = cv2.imread("D:/tensorflow/test/2.jpg")
        image = cv2.imread(imgList[i])
        start = cv2.getTickCount()
        image_np_expanded = np.expand_dims(image,axis = 0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        (boxes,scores,classes,num_detections) = sess.run([boxes,scores,classes,\
                                num_detections],feed_dict={image_tensor:image_np_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
          image,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          categorys_index,
          min_score_thresh = 0.4,
          use_normalized_coordinates = True,
          line_thickness = 4  
          )
        end = cv2.getTickCount()
        use_time = (end - start) / cv2.getTickFrequency()
        print("use_time:%0.3fs" % use_time)
        if num < 10:
            winName = "img%d"%num
            image = cv2.resize(image,(image.shape[1],image.shape[0]))
            cv2.imshow(winName, image)
        else:
            break
cv2.waitKey(0)
cv2.destroyAllWindows()
