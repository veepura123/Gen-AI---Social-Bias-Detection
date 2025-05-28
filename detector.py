import streamlit as st
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

# Load the tokenizer from the fine-tuned model directory
tokenizer = BertTokenizer.from_pretrained('./fine_tune_model_tf')

# Recreate the model architecture for the fine-tuned model
class BiasDetectionModel(tf.keras.Model):
    def __init__(self, bert_model):
        super(BiasDetectionModel, self).__init__()
        self.bert = bert_model
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.classifier = tf.keras.layers.Dense(11, activation='softmax')  # Adjusted to 11 categories

    def call(self, inputs):
        outputs = self.bert(inputs)[0]
        pooled_output = outputs[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

# Load the base BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Instantiate the model
model = BiasDetectionModel(bert_model)

# Compile the model with fine-tuned settings
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Load the fine-tuned weights
model.load_weights('./fine_tune_model_tf/model_weights')

# Function for inference
@st.cache_data
def predict_bias(text):
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="tf")
    outputs = model(inputs['input_ids'])
    return outputs.numpy()[0]

# Streamlit app
st.title('Bias Detection App (Fine-tuned Model)')

# Text input
user_input = st.text_area("Enter text to analyze for bias:", "")

if st.button('Detect Bias'):
    if user_input:
        predictions = predict_bias(user_input)
        bias_categories = ['Caste', 'Religion', 'Age', 'Disability', 'Gender', 
                           'Physical Appearance', 'Socioeconomic', 'Race-Color',
                           'Nationality', 'Sexual Orientation', 'Religion']
        
        st.subheader('Bias Detection Results:')
        for category, prob in zip(bias_categories, predictions):
            st.write(f"{category}: {prob:.4f}")
            st.progress(float(prob))
    else:
        st.warning('Please enter some text to analyze.')

st.markdown("---")
st.markdown("This app uses a fine-tuned model to detect potential biases in text across multiple categories.")
st.markdown("A higher score indicates a higher likelihood of bias in that category.")
