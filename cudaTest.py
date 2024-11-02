import tensorflow as tf

# Check if TensorFlow can detect GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Check details about the detected GPUs (if any)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"Device found: {gpu}")
else:
    print("No GPU devices found!")
