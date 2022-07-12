
0、参考文章
https://techvidvan.com/tutorials/hand-gesture-recognition-tensorflow-opencv/

==========================


1、建立环境
conda create --name GestureRecognition python=3.8 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

==========================

2、切换环境
conda activate GestureRecognition

==========================

3、安装必要的包
3.1 安装open-cv
pip install opencv-python

3.2 安装mediapipe
pip install mediapipe

3.3 安装tf
pip install tensorflow

3.4 下载预训练好的文件
https://techvidvan.s3.amazonaws.com/machine-learning-projects/hand-gesture-recognition-code.zip

==========================

4、引入必要的包
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

5、初始化mp

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils


6、load模型

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')
# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

================

OK，看下来有几个姿态哈
2022-07-12 20:47:06.133029: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']
(GestureRecognition) lemonhall@yuningdeMBP:~/gestureRecognition$


也不多

okay
peace
thumbs up
thumbs down
call me
stop
rock
live long
fist
smile

gesture.names确实就是一个以回车分隔的类名文件
classNames = f.read().split('\n')
['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']
这个小技巧很好，记下来

================

# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)

while True:
  # Read each frame from the webcam
  _, frame = cap.read()
  x , y, c = frame.shape

  # Flip the frame vertically
  frame = cv2.flip(frame, 1)
  # Show the final output
  cv2.imshow("Output", frame)
  if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

接下来是用opencv打开摄像头

=================

接着是拿到所有的关键点

  framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # Get hand landmark prediction
  result = hands.process(framergb)
  className = ''

  # post process the result
  if result.multi_hand_landmarks:
  	landmarks = []
  	for handslms in result.multi_hand_landmarks:
  		for lm in handslms.landmark:# print(id, lm)
  			lmx = int(lm.x * x)
			lmy = int(lm.y * y)
			landmarks.append([lmx, lmy])
			# Drawing landmarks on frames
			mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

===这段话是这样的

#这段代码是这样的，第一次会失败，因为mac会问一下权限问题，第二次ok
#按下q会结束程序
while True:
	# Read each frame from the webcam
	_, frame = cap.read()
	x,y,c = frame.shape

	# Flip the frame vertically
	frame = cv2.flip(frame, 1)
	framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# Get hand landmark prediction
	result = hands.process(framergb)

	className = ''

	# post process the result
	if result.multi_hand_landmarks:
		landmarks = []
		for handslms in result.multi_hand_landmarks:
			for lm in handslms.landmark:
				lmx = int(lm.x*x)
				lmy = int(lm.y*y)
				landmarks.append([lmx,lmy])
			mpDraw.draw_landmarks(frame,handslms,mpHands.HAND_CONNECTIONS)
	cv2.imshow("Output", frame)
	if cv2.waitKey(1) == ord('q'):
		break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()

但是可能是因为是教程，所以其实完整版是这样的



