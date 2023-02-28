import cv2 
import numpy as np 
import argparse 
import time 

# # show_img = cv2.imread('img/person.jpg')
# # cv2.imshow('img',show_img)
# # cv2.waitKey(0)


# # Read in an image
# # img = cv2.imread('img/person.jpg')
# # cv2.imshow('Park', img)

# # # Converting to grayscale
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # cv2.imshow('Gray', gray)

# # # Blur 
# # blur = cv2.GaussianBlur(img, (7,7), cv2.BORDER_DEFAULT)
# # cv2.imshow('Blur', blur)

# # # Edge Cascade
# # canny = cv2.Canny(blur, 125, 175)
# # cv2.imshow('Canny Edges', canny)

# # # Dilating the image
# # dilated = cv2.dilate(canny, (7,7), iterations=3)
# # cv2.imshow('Dilated', dilated)

# # # Eroding
# # eroded = cv2.erode(dilated, (7,7), iterations=3)
# # cv2.imshow('Eroded', eroded)

# # # Resize
# # resized = cv2.resize(img, (500,500), interpolation=cv2.INTER_CUBIC)
# # cv2.imshow('Resized', resized)

# # # Cropping
# # cropped = img[50:200, 200:400]
# # cv2.imshow('Cropped', cropped)




# load yolo file 

def load_yolo():
	net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")
	classes = []
	with open("./coco.names", "r") as f:
		classes = [line.strip() for line in f.readlines()] 
	
	output_layers = [layer_name for layer_name in net.getUnconnectedOutLayersNames()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))
	return  net, classes, colors,output_layers
net, classes, colors,output_layers = (load_yolo())


#--------------------------------------------------------------------------------------------------------

#read image and extract height and width 

def load_image(img_path):
	img = cv2.imread(img_path)
	img = cv2.resize(img, None, fx=0.9, fy=0.9)
	height, width, channels = img.shape
	return channels,height, width,img
channels,height, width,img = load_image('img\person.jpg')

#-------------------------------------------------------------------------------------------------------- 

# detect obj 

def detect_objects(img, net, output_layers):			
	blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
	net.setInput(blob)
	outputs = net.forward(output_layers)
	return blob, outputs

blob, outputs = detect_objects(img, net, output_layers)


#--------------------------------------------------------------------------------------------------------------- 

#find box dimensions 

def get_box_dimensions(outputs, height, width):
	boxes = []
	confs = []
	class_ids = []
	for output in outputs:
		for detect in output:
			scores = detect[5:]
			class_id = np.argmax(scores)
			conf = scores[class_id]
			if conf > 0.3:
				center_x = int(detect[0] * width)
				center_y = int(detect[1] * height)
				w = int(detect[2] * width)
				h = int(detect[3] * height)
				x = int(center_x - w/2)
				y = int(center_y - h / 2)
				boxes.append([x, y, w, h])
				confs.append(float(conf))
				class_ids.append(class_id)
	return boxes, confs, class_ids
boxes, confs, class_ids =get_box_dimensions(outputs, height, width)


#------------------------------------------------------------------------------------------------------------------ 

#draw box for obj 


def draw_labels(boxes, confs, colors, class_ids, classes, img): 
	indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
	font = cv2.FONT_HERSHEY_PLAIN
	for i in range(len(boxes)):
		if i in indexes:
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			color = colors[i]
			cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
			cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
	cv2.imshow("Image", img)


draw_labels(boxes, confs, colors, class_ids, classes, img)

cv2.waitKey(0)





