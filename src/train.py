import os
import argparse
from model import build_model
from data_loader import get_data_generators
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train(data_dir, epochs=20, batch_size=32, model_save_path='models/tongue_disease_model.h5'):
    # Create models directory if not exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_gen, val_gen = get_data_generators(data_dir, batch_size=batch_size)
    
    if train_gen is None or train_gen.samples == 0:
        print("No data found. Please populate the data directory.")
        return
    
    num_classes = train_gen.num_classes
    print(f"Detected {num_classes} classes: {list(train_gen.class_indices.keys())}")
    
    # Build model
    print("Building model...")
    model = build_model(num_classes=num_classes)
    
    # Callbacks
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train
    print("Starting training...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[checkpoint, early_stop]
    )
    
    print("Training finished.")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the Tongue Disease Detection Model')
    parser.add_argument('--data_dir', type=str, default='data/raw', help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    train(args.data_dir, args.epochs, args.batch_size)
