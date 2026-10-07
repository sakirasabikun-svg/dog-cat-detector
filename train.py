import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# 1. Load the Data
data_dir = 'dataset'
img_height = 150
img_width = 150
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir, validation_split=0.2, subset="training", seed=123, image_size=(img_height, img_width), batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir, validation_split=0.2, subset="validation", seed=123, image_size=(img_height, img_width), batch_size=batch_size)

# 2. Data Augmentation
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.1),
])

# 3. THE MAGIC: Borrowing Google's Pre-trained Brain (MobileNetV2)
# 'imagenet' means it already knows millions of real-world images!
base_model = MobileNetV2(input_shape=(img_height, img_width, 3),
                         include_top=False,
                         weights='imagenet')

# We lock this brain so we don't accidentally erase its huge memory
base_model.trainable = False 

# 4. Build the Final Super-Smart Model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = preprocess_input(x) # This layer translates the image into the exact logic the Google brain understands
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(2)(x) # 2 choices: Dog or Cat

model = tf.keras.Model(inputs, outputs)

# 5. Compile the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("\n--- Starting Super Smart Training (Transfer Learning) ---")
print("Because the brain is already smart, we only need 5 Epochs!\n")

# Only 5 epochs are needed now!
epochs = 5 
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# 6. Save the Model
model.save('dog_cat_model.keras')
print("✅ Super Smart model saved successfully as: dog_cat_model.keras")