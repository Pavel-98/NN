# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf

print(tf.__version__)
#In[]:

EPOCHS = 1
# GRADED FUNCTION: train_mnist
def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE

    # YOUR CODE SHOULD END HERE

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # YOUR CODE SHOULD START HERE

    # YOUR CODE SHOULD END HERE

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
        # YOUR CODE HERE,
        # YOUR CODE HERE,
        # YOUR CODE HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    history = model.fit(x_train, y_train, epochs=EPOCHS  # YOUR CODE HERE
    )
    # model fitting

    return history.epoch, history.history['accuracy'][-1]

while EPOCHS <= 10:
    result = train_mnist()
    if result[1] > 0.99:
        print("Reached 99% accuracy so cancelling training!")
        break
    EPOCHS += 1








def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
    #print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
