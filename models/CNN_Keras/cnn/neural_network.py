from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense



class CNN:
    @staticmethod
    def build(width, height, depth, total_classes, Saved_Weights_Path=None):
        # Initialize the Model
        model = Sequential()

        # First CONV => RELU => POOL Layer
        model.add(Conv2D(20, (5, 5), padding="same", input_shape=(height, width, depth)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"))

        # Second CONV => RELU => POOL Layer
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"))

        # Third CONV => RELU => POOL Layer
        # Convolution -> ReLU Activation Function -> Pooling Layer
        model.add(Conv2D(100, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format="channels_last"))

        # FC => RELU layers
        #  Fully Connected Layer -> ReLU Activation Function
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # Using Softmax Classifier for Linear Classification
        model.add(Dense(total_classes))
        model.add(Activation("softmax"))

        # If the saved_weights file is already present i.e model is pre-trained, load that weights
        if Saved_Weights_Path is not None:
            model.load_weights(Saved_Weights_Path)
        return model
# --------------------------------- EOC ------------------------------------
