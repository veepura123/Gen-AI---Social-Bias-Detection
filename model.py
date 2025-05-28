import tensorflow as tf
from tensorflow import keras
from transformers import TFBertForSequenceClassification, BertTokenizer, TFBertModel
import pandas as pd
import numpy as np

from utils import load_sbic_dataset, load_validation_dataset

# Load datasets
train_texts, train_labels, test_texts, test_labels = load_sbic_dataset(
    train_path="datasets/SBIC_train_processed.csv",
    test_path="datasets/SBIC_test_processed.csv",
)
val_texts, val_labels = load_validation_dataset(
    dev_path="datasets/SBIC_dev_processed.csv"
)

# Load tokenizer and base BERT model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = TFBertModel.from_pretrained(model_name)

# Define custom model for multi-label classification
class BiasDetectionModel(tf.keras.Model):
    def __init__(self, bert_model):
        super(BiasDetectionModel, self).__init__()
        self.bert = bert_model
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(7, activation='sigmoid')

    def call(self, inputs):
        outputs = self.bert(inputs)[0]
        pooled_output = outputs[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# Instantiate the model
model = BiasDetectionModel(bert_model)

# Compile the model
optimizer = keras.optimizers.Adam(learning_rate=2e-5)
loss = keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])

# Tokenize and encode the datasets
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="tf")

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).shuffle(1000).batch(16)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
)).batch(16)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=3  # Adjust as needed
)

# Save the model
model.save_weights('./bias_detection_model_tf/model_weights')
tokenizer.save_pretrained('./bias_detection_model_tf')

# Function for inference
def predict_bias(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="tf")
    outputs = model(inputs)
    return outputs.numpy()[0]

# Example prediction
example_text = "This is an example text to test bias prediction."
predictions = predict_bias(example_text)
bias_categories = ['body', 'culture', 'disabled', 'gender', 'race', 'social', 'victim']
for category, prob in zip(bias_categories, predictions):
    print(f"{category}: {prob:.4f}")