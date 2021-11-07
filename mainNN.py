import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)

def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs.get('accuracy') is not None and logs.get('accuracy') >= 0.99:
                print("\nReached 99% accuracy so cancelling training!\n")
                self.model.stop_training = True
    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # YOUR CODE SHOULD START HERE
    input_shape=(28, 28, 1)
    num_classes = 10
    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    history = model.fit(x=x_train, y=y_train,epochs = 210, callbacks=[myCallback()]  # YOUR CODE HERE
                        )
    # model fitting

    return history.epoch, history.history['accuracy'][-1]
train_mnist()


