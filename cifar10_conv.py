import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

(x_train,y_train), (x_test,y_test) = datasets.cifar10.load_data()

x_train = tf.cast(x_train,dtype=tf.float32)/255.0
y_train = tf.cast(y_train,dtype=tf.int32)
x_test = tf.cast(x_test,dtype=tf.float32)/255.0
y_test = tf.cast(y_test,dtype=tf.int32)

conv_layers = [
	layers.Conv2D(64, kernel_size=[3,3], padding="same", activation=tf.nn.relu),
	layers.Conv2D(64, kernel_size=[3,3], padding="same", activation=tf.nn.relu),
	layers.MaxPool2D(pool_size=2, strides=2, padding="same"),
	layers.Conv2D(128, kernel_size=[3,3], padding="same", activation=tf.nn.relu),
	layers.Conv2D(128, kernel_size=[3,3], padding="same", activation=tf.nn.relu),
	layers.MaxPool2D(pool_size=2, strides=2, padding="same"),
	layers.Flatten(),
	layers.Dense(64, activation=tf.nn.relu),
	layers.Dense(32, activation=tf.nn.relu),
	layers.Dense(10, activation='softmax')
]

model = Sequential(conv_layers)
model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.fit(x=x_train, y=y_train, epochs=50, validation_data=(x_test, y_test))