import streamlit as st
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
import numpy as np

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('./bias_detection_model_tf')

# Recreate the model architecture
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

# Load the base BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Instantiate the model
model = BiasDetectionModel(bert_model)

# Compile the model (necessary before loading weights)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['binary_accuracy']
)

# Load the trained weights
model.load_weights('./bias_detection_model_tf/model_weights')

# Function for inference
@st.cache_data
def predict_bias(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="tf")
    outputs = model(inputs)
    return outputs.numpy()[0]

# Streamlit app
st.title('Bias Detection App')

# Text input
user_input = st.text_area("Enter text to analyze for bias:", "")

if st.button('Detect Bias'):
    if user_input:
        predictions = predict_bias(user_input)
        bias_categories = ['body', 'culture', 'disabled', 'gender', 'race', 'social', 'victim']
        
        st.subheader('Bias Detection Results:')
        for category, prob in zip(bias_categories, predictions):
            st.write(f"{category.capitalize()}: {prob:.4f}")
            st.progress(float(prob))
    else:
        st.warning('Please enter some text to analyze.')

st.markdown("---")
st.markdown("This app detects potential biases in text across multiple categories.")
st.markdown("A higher score indicates a higher likelihood of bias in that category.")
