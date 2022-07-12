# import necessary packages for hand gesture recognition project using Python OpenCV
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from threading import Thread
import sys
from queue import Queue
import time

#1、建立环境
#conda create --name GestureRecognition python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

#2、切换环境
#conda activate GestureRecognition

#3、安装必要的包
#3.1 安装open-cv
#pip install opencv-python

#3.2 安装mediapipe
#pip install mediapipe

#3.3 安装tf
#pip install tensorflow

#3.4 下载预训练好的文件
#https://techvidvan.s3.amazonaws.com/machine-learning-projects/hand-gesture-recognition-code.zip

#接下来就是开始编码了


#晚上，我修改了第二版本，使用一个单独的线程来做计算和frame上的字体处理啥的
#延迟已经很低了，可以接受了，但即使是这样cpu还是看到手的时候就飙到90%左右，
#心累啊，说到底，就是.....哀家需要GPU......
#行吧行吧，以后得配一个GPU的运算机器啊
class FileVideoStream:
	def __init__(self, path, queueSize=128):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path,cv2.CAP_FFMPEG)
		self.stopped = False

		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queueSize)
		print("FileVideoStream has been INIT")

	def start(self):
		# start a thread to read frames from the file video stream
		t = Thread(target=self.update, args=())
		print("FileVideoStream has been started")
		t.daemon = True
		t.start()
		return self

	def update(self):
		counter = 0
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				print("the thread indicator variable is set, stop the threading")
				return

			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()

				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					print("I am not grabbed.....")
					self.stop()
					return

				#用这个跳帧法，基本能做到CPU50%不发烫了，好不容易啊
				#所以还是需要GPU，有GPU了，我感觉，可能CPU能降低到5-10，能耗再降低很多
				if counter == 2:
					counter = 0
				else:
					x,y,c = frame.shape
					# Flip the frame vertically
					# frame = cv2.flip(frame, 1)
					framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

					# Get hand landmark prediction
					result = hands.process(framergb)
					className = ''
					landmarks = []

					# post process the result
					if result.multi_hand_landmarks:
						for handslms in result.multi_hand_landmarks:
							for lm in handslms.landmark:
								lmx = int(lm.x*x)
								lmy = int(lm.y*y)
								landmarks.append([lmx,lmy])

					# Predict gesture in Hand Gesture Recognition project
					if landmarks:
						prediction = model.predict([landmarks])
						#print(prediction)
						classID = np.argmax(prediction)
						className = classNames[classID]
						# show the prediction on the frame
						cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2, cv2.LINE_AA)

				# add the frame to the queue
				self.Q.put(frame)
				#print("fvs.update has been called and Q size is:"+str(self.Q.qsize()))
			else:
				print("the queue is full of data")
			counter = counter +1

	def read(self):
		# return next frame in the queue
		# print("fvs.read has been called and Q size is:"+str(self.Q.qsize()))
		return self.Q.get()

	def more(self):
		# return True if there are still frames in the queue
		return True

	def stop(self):
		# indicate that the thread should be stopped
		print("thread stop method has been called")
		self.stopped = True


# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')
# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Initialize the webcam for Hand Gesture Recognition Python project
#cap = cv2.VideoCapture(0)

#参考文章：https://lindevs.com/capture-rtsp-stream-from-ip-camera-using-opencv/
#等于说就是换了下面那三句话而已
RTSP_URL = 'rtsp://192.168.50.221:8554/live'
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
#cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

fvs = FileVideoStream(RTSP_URL).start()
time.sleep(1.0)

while fvs.more():
	frame = fvs.read()
	cv2.imshow("Output", frame)
	if cv2.waitKey(1) == ord('q'):
 		break
cv2.destroyAllWindows()