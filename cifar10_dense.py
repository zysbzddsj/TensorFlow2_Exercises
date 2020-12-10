import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

(x,y), (x_test,y_test) = datasets.cifar10.load_data()

x = 2*tf.cast(x,dtype=tf.float32)/255.0-1.0
y = tf.cast(y,dtype=tf.int32)
y = tf.squeeze(y)

x_test = 2*tf.cast(x_test,dtype=tf.float32)/255.0-1.0
y_test = tf.cast(y_test,dtype=tf.int32)
y_test = tf.squeeze(y_test)
y = tf.one_hot(y, depth = 10)
y_test = tf.one_hot(y_test, depth = 10)
print('dataset', x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(10000).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.batch(128)

class MyDense(layers.Layer):
	
	def __init__(self, inp_dim, out_dim):
		super(MyDense, self).__init__()                                                                                                                          
		self.kernel = self.add_weight('w', [inp_dim, out_dim])
	
	def call(self, inputs, training = None):
		a = inputs @ self.kernel
		return a
		
class MyNetwork(keras.Model):

	def __init__(self):
		super(MyNetwork, self).__init__()
		self.fc1 = MyDense(32*32*3, 256)
		self.fc2 = MyDense(256, 128)
		self.fc3 = MyDense(128, 64)
		self.fc4 = MyDense(64, 32)
		self.fc5 = MyDense(32, 10)
	
	def call(self, inputs, training = None):
		a0 = tf.reshape(inputs, [-1,32*32*3])
		a1 = self.fc1(a0)
		a1 = tf.nn.relu(a1)
		a2 = self.fc2(a1)
		a2 = tf.nn.relu(a2)
		a3 = self.fc3(a2)
		a3 = tf.nn.relu(a3)
		a4 = self.fc4(a3)
		a4 = tf.nn.relu(a4)
		a5 = self.fc5(a4)
		return a5

network = MyNetwork()
network.build(input_shape = [None, 32*32*3])
network.compile(optimizer=optimizers.Adam(lr=1e-3),
				loss=tf.losses.CategoricalCrossentropy(from_logits=True),
				metrics=['accuracy'])
network.fit(train_db, epochs=15, validation_data=test_db, validation_freq=1)

		
