

# Imports
import numpy as np
import random
import tensorflow as tf
import ObjectTracking_utils as utils
from tensorflow.contrib import rnn
import cv2
import os.path
import time

class ObjectTracking:
    # Tracker Network Parameters
    rolo_weights_file = '/home/mdeshpa3/Tracker/model_step6_exp1.ckpt' 
    num_steps = 6  # number of frames as an input sequence
    num_feat = 4096
    num_predict = 6 # final output of LSTM 6 loc parameters
    num_gt = 4 # groundtruth for x,y,width and height
    num_input = num_feat + num_predict # data input: 4096+6= 4102 (number of features obtained from YOLO + number of predicted locations)
    hidden_size = 2*num_input #= 8204
    # Tracker Training Parameters
    learning_rate = 0.00001 #training
    training_iters = 210#100000
    batch_size = 1 #128
    display_step = 1
    num_layers = 3
    dropout = dropout = 0.35

    # tf Graph input
    input_x = tf.placeholder("float32", [None, num_steps, num_input])
    initstate = tf.placeholder("float32", [None, 2*num_input]) #state & cell => 2x num_input
    input_y = tf.placeholder("float32", [None, num_gt])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([num_input, num_predict]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([num_predict]))
    }

    def __init__(self):
        print("init Object Tracking training")
        self.TrackerModel()


    '''def BuildLSTM(self, name,  input_x, initstate, weights, biases,reuse):
	with tf.variable_scope("LSTM") as scope:
		if (reuse):
            		tf.get_variable_scope().reuse_variables()
		#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
		input_x = tf.unstack(input_x ,self.num_steps,axis = 1)	
		cell = rnn.LSTMBlockCell(self.num_input,forget_bias=1)
		outputs,states=rnn.static_rnn(cell,input_x,dtype="float32")
		reuse = True 
		tf.get_variable_scope().reuse_variables()

		return outputs'''

    def BuildLSTM(self, name,  input_x, initstate, weights, biases,reuse):
	with tf.variable_scope("LSTM") as scope:
		if (reuse):
            		tf.get_variable_scope().reuse_variables()
		#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
		#input_x = tf.unstack(input_x ,self.num_steps,axis = 1)	
		cells = []
		for _ in range(self.num_layers):
			cell = rnn.LSTMBlockCell(self.hidden_size,forget_bias=1)
			cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - self.dropout)
	  		cells.append(cell)
		cell = tf.contrib.rnn.MultiRNNCell(cells)  #RNN cell composed sequentially of multiple simple cells.

		outputs,states=tf.nn.dynamic_rnn(cell,input_x,dtype="float32") #dynamic unrolling of inputs
		reuse = True 
		tf.get_variable_scope().reuse_variables()

		return outputs


    # Experiment with dropout
    def dropout_features(self, feature, prob):
        num_drop = int(prob * 4096)
        drop_index = random.sample(xrange(4096), num_drop)
        for i in range(len(drop_index)):
            index = drop_index[i]
            feature[index] = 0
        return feature

    def build_LSTM_networks(self):
	print "Building LSTM Network"

        self.lstm_module = self.BuildLSTM('lstm_test', self.input_x, self.initstate, self.weights, self.biases,False)
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.saver = tf.train.Saver()
        print "Loading complete!" + '\n'

    def training(self):
        print("TRAINING Object Tracker...")
        log_file = open("Output/training-20-log.txt", "a") #open in append mode
        self.build_LSTM_networks()
        num_videos = 20
        epoches = 20 * 100   

	'''Tuning Object Trackeing Model'''    
        predict = self.BuildLSTM('lstm_train', self.input_x, self.initstate, self.weights, self.biases,True)
        self.predict_location = predict[0][:, 4097:4101]
	#Find Mean square error
        self.correct_prediction = tf.square(self.predict_location - self.input_y)
        self.accuracy = tf.reduce_mean(self.correct_prediction) * 100
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.accuracy) # Adam Optimizer
        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(epoches):
                i = epoch % num_videos
                [self.w_img, self.h_img, sequence_name, dummy, self.training_iters]= utils.choose_video_sequence(i)
		
                x_path = os.path.join('/home/mdeshpa3/Tracker/Data', sequence_name, 'yolo_out/')
                y_path = os.path.join('/home/mdeshpa3/Tracker/Data', sequence_name, 'groundtruth_rect.txt')
                self.output_path = os.path.join('/home/mdeshpa3/Tracker/DATA', sequence_name, 'rolo_out_train/')
                utils.createFolder(self.output_path)
		print sequence_name,x_path
                total_loss = 0
                id = 0
                # Keep training until reach max iterations
                while id  < self.training_iters- self.num_steps:
                    # Load training data & ground truth
                    batch_input = self.rolo_utils.load_yolo_output_test(x_path, self.batch_size, self.num_steps, id) # [num_of_examples, num_input] (depth == 1)

                    # Apply dropout to batch_xs
                    #for item in range(len(batch_xs)):
                    #    batch_xs[item] = self.dropout_features(batch_xs[item], 0.4)

                    batch_groundtruth = self.rolo_utils.load_rolo_gt_test(y_path, self.batch_size, self.num_steps, id)
                    batch_groundtruth = utils.locations_from_0_to_1(self.w_img, self.h_img, batch_groundtruth)

                    # Reshape data to get 3 seq of 5002 elements
                    batch_input = np.reshape(batch_input, [self.batch_size, self.num_steps, self.num_input])
                    batch_groundtruth = np.reshape(batch_groundtruth, [self.batch_size, 4])

                    predict_location= sess.run(self.predict_location,feed_dict={self.input_x: batch_input, self.input_y: batch_groundtruth, self.initstate: np.zeros((self.batch_size, 2*self.num_input))})
       
                    #print("len(pred) = ", len(pred_location))
                    print("ROLO Pred in pixel: ", predict_location[0][0]*self.w_img, predict_location[0][1]*self.h_img, predict_location[0][2]*self.w_img, predict_location[0][3]*self.h_img)
                  
                    # Save pred_location to file
                    utils.save_rolo_output(self.output_path, predict_location, id, self.num_steps, self.batch_size)

                    sess.run(self.optimizer, feed_dict={self.input_x: batch_input, self.input_y: batch_groundtruth, self.initstate: np.zeros((self.batch_size, 2*self.num_input))})
                    if id % self.display_step == 0:
                        # Calculate batch loss
                        loss = sess.run(self.accuracy, feed_dict={self.input_x: batch_input, self.input_y: batch_groundtruth, self.initstate: np.zeros((self.batch_size, 2*self.num_input))})
                        total_loss += loss
                    id += 1

                #print "Optimization Finished!"
                avg_loss = total_loss/id
                print "Avg loss: " + sequence_name + ": " + str(avg_loss)

                log_file.write(str("{:.3f}".format(avg_loss)) + '  ')
                if i+1==num_videos:
                    log_file.write('\n')
                    save_path = self.saver.save(sess, self.rolo_weights_file)
                    print("Model saved in file: %s" % save_path)

        log_file.close()
        return


    def TrackerModel(self):

            self.rolo_utils= utils.ROLO_utils()
            self.training()

def main():
        ObjectTracking()

if __name__=='__main__':
        main()
