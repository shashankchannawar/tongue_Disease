import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input

def build_model(num_classes, input_shape=(224, 224, 3)):
    """
    Builds a CNN model using MobileNetV2 for transfer learning.
    
    Args:
        num_classes (int): Number of disease categories.
        input_shape (tuple): Shape of input images.
        
    Returns:
        model: Compiled Keras model.
    """
    # Base model with pre-trained ImageNet weights
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = build_model(num_classes=3)
    model.summary()
