class VGGNet():
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential([
        Conv2D(64, (3, 3), input_shape=(width,height,depth), padding='same',
               activation='relu'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same',),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            
        Conv2D(256, (3, 3), activation='relu', padding='same',),
        Conv2D(256, (3, 3), activation='relu', padding='same',),
        Conv2D(256, (3, 3), activation='relu', padding='same',),

        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),

        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),
        Conv2D(512, (3, 3), activation='relu', padding='same',),

        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.4),
        Dense(4096, activation='relu'),
        Dropout(0.4),
        Dense(1000, activation='relu'),
        Dropout(0.4),
        Dense(classes, activation='softmax')
        ])
        return model
