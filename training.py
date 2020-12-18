from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.losses import categorical_crossentropy

def get_model(inp_shape, num_class):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=inp_shape))
    model.add(Conv2D(32, kernel_size=(3,3), activate='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dropout(0.5))
    model.add(Dense(num_class, activate='softmax'))

    model.compile(loss=categorical_crossentropy, optimizer="Adam", metrics=['accuracy'])
    print(model.summary())

    return model
