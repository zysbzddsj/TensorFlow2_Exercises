import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

(x,y),_ = datasets.mnist.load_data()

x = tf.convert_to_tensor(x,dtype=tf.float32)/255.0
y = tf.convert_to_tensor(y,dtype=tf.int32)

# print(x.shape,y.shape,x.dtype,y.dtype) 
# print(tf.reduce_min(x),tf.reduce_max(x))
# print(tf.reduce_min(y),tf.reduce_max(y))

# sample&batch
train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
# print('batch:', sample[0].shape, sample[1].shape)

# 784 -> 256 -> 128 -> 10
w1 = tf.Variable(tf.random.truncated_normal([784,256],stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256,128],stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))
lr = 0.0005

# h1=x@w1+b1
for i in range(10):
    for step,(x,y) in enumerate(train_db):   
	    x = tf.reshape(x, [-1, 28*28])
	    with tf.GradientTape() as tape:
	        h1 = x@w1 + b1  
	        h1 = tf.nn.relu(h1)    
	        h2 = h1@w2 + b2    
	        h2 = tf.nn.relu(h2)    
	        out = h2@w3 + b3
	
	        #loss
	        y_onehot = tf.one_hot(y, depth = 10)
	        loss = tf.square(y_onehot-out)
	        loss = tf.reduce_mean(loss)
	
	    #compute gradient
	    grad = tape.gradient(loss, [w1,b1,w2,b2,w3,b3])
	    w1 = w1 - lr*grad[0]
	    w1 = tf.Variable(w1)
	    b1 = b1 - lr*grad[1]
	    b1 = tf.Variable(b1)
	    w2 = w2 - lr*grad[2]
	    w2 = tf.Variable(w2)
	    b2 = b2 - lr*grad[3]
	    b2 = tf.Variable(b2)
	    w3 = w3 - lr*grad[4]
	    w3 = tf.Variable(w3)
	    b3 = b3 - lr*grad[5]
	    b3 = tf.Variable(b3)
	
	    if step%200 == 0:
	        print(i,step, 'loss', float(loss))
	