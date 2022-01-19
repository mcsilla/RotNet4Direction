from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

class RotNet4:
    def __init__(self, size):

        self.size = size

        # number of classes
        nb_classes = 4

        # input image shape
        input_shape = (self.size, self.size, 3)

        # load base model
        base_model = ResNet50(weights='imagenet', include_top=False,
                              input_shape=input_shape)
        # append classification layer
        x = base_model.output
        x = Flatten()(x)
        final_output = Dense(nb_classes, activation='softmax', name='fc4')(x)

        # create the new model
        self.model = Model(inputs=base_model.input, outputs=final_output)

        # self.model.summary()




