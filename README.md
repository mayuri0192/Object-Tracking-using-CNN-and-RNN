Object Tracking using CNN and RNN. 

Followed the paper : https://arxiv.org/pdf/1607.05781.pdf

Dataset: http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html


Designed a robust tracker which tracks appearance and motion over time.
Recurrent Neural Network (LSTMs) tracks the temporal information of the object and Convolutional Neural Network
(using YOLO) for Object Classification and bounding box prediction


The 4096 visual features obtained from YOLO Convolutional layer is concatenated with the 6 location coordinates of bounding boxes obtained after the fully connected layer of YOLO. 

These 4096 + 6 = 4102 features are given to stacked LSTM as input. 

With the help of visual features of the objects, the next location of the bounding boxes is predicted by the LSTM.
