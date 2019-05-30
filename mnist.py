print('MNIST initializing...')

def teste():
    return 'teste'

def run(epochs=0, name='simple-mnist-cnn'):
    import numpy

    import keras
    from keras.datasets import mnist

    print('========================================================')

    # loads the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # printing the number of samples in x_train, x_test, y_train, y_test

    print("Initial shape or dimensions of x_train" + str(x_train.shape))

    print()

    print("Number of samples in our training data:" + str(len(x_train)))
    print("Number of labels in our training data:" + str(len(y_train)))

    print()

    print("Number of samples in our testing data:" + str(len(x_test)))
    print("Number of labels in our testing data:" + str(len(y_test)))

    print()

    print("Dimensions of x_train:" + str(x_train[0].shape))
    print("Labels in y_train:" + str(y_train.shape))

    print()

    print("Dimensions of x_test:" + str(x_test[0].shape))
    print("Labels in y_test:" + str(y_test.shape))

    # # # # Using OpenCV

    # # # # pip install opencv-python
    # # # # pip install opencv-contrib-python
    # # # import cv2

    # # # # Use OpenCV to display 6 random images from our dataset
    # # # for i in range(0, 6):
    # # #     random_num = numpy.random.randint(0, len(x_train))
    # # #     img = x_train[random_num]
    # # #     window_name = 'Random Sample #' + str(i)
    # # #     cv2.imshow(window_name, img)
    # # #     cv2.waitKey(0)
    # # #     cv2.destroyWindow(window_name)

    # # # cv2.destroyAllWindows()

    # # # # Using matplotlib

    # # # import matplotlib.pyplot as plt

    # # # # Plots 6 images, note sublplot's arguments are nrows, ncols, index
    # # # # we set the color map to grey since our image dataset is grayscale

    # # # plt.subplot(331)
    # # # random_num = numpy.random.randint(0,len(x_train))
    # # # plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

    # # # plt.subplot(332)
    # # # random_num = numpy.random.randint(0,len(x_train))
    # # # plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

    # # # plt.subplot(333)
    # # # random_num = numpy.random.randint(0,len(x_train))
    # # # plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

    # # # plt.subplot(334)
    # # # random_num = numpy.random.randint(0,len(x_train))
    # # # plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

    # # # plt.subplot(335)
    # # # random_num = numpy.random.randint(0,len(x_train))
    # # # plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

    # # # plt.subplot(336)
    # # # random_num = numpy.random.randint(0,len(x_train))
    # # # plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

    # # # plt.show()

    print('========================================================')

    # Lets store the number of rows and columns
    img_rows = x_train[0].shape[0]
    img_cols = x_train[1].shape[0]

    # Getting our data in the right 'shape' for Keras
    # We need to add a 4th dimension to our data thereby changing our
    # original image shape of (60000, 28, 28) to (60000, 28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)

    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    # Store the shape of a single image
    input_shape = (img_rows, img_cols, 1)

    # Change our image type to float32 data type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalize our data by changing the range from (0 to 255) to (0 to 1)
    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    print('========================================================')

    from keras.utils import np_utils

    # Now we one hot enconde outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Let's count the number columns in our hot encoded matrix
    print('Number of classes: ' + str(y_test.shape[1]))

    num_classes = y_test.shape[1]
    num_pixels = x_train.shape[1] * x_train.shape[2]

    print('========================================================')

    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
    from keras.optimizers import SGD

    # create model
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
            activation='relu' ,
            input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy',
                optimizer = SGD(0.01),
                metrics = ['accuracy'])

    print(model.summary())

    print('========================================================')

    batch_size = 32

    history = model.fit(x_train,
                        y_train,
                        batch_size=batch_size,
                        epochs=int(epochs),
                        verbose=1,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    print('========================================================')

    # Plotting our loss charts

    import matplotlib.pyplot as plt

    history_dict = history.history

    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    # acc_values = history_dict['acc']
    # val_acc_values = history_dict['val_acc']
    epochs = range(1, len(loss_values) + 1)

    line1 = plt.plot(epochs, val_loss_values, label='Validation/Test Loss')
    line2 = plt.plot(epochs, loss_values, label='Training Loss')
    plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
    plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    import os 
    model_name = os.getcwd() + '\\' + name + '.h5'
    model.save(model_name)
    print('Model saved at ' + model_name)

    return history_dict