import numpy as np
import tensorflow as tf

# Hyperparameters (you can adjust these)
vocab_size = 1000  # Approximate number of unique words
embedding_dim = 128  # Dimension of word embeddings
units = 512  # Number of units in the LSTM layer
epochs = 20  # Number of training epochs

# Sample text data (replace with your actual data)
text = "This is a sample text used for training the language model. It can be a paragraph, a story, or any collection of text you want to use."

# Preprocess the text
def preprocess_text(text):
  # Tokenize the text (you can choose word-level or character-level)
  tokens = text.lower().split()  # Example of word-level tokenization
  # Create a vocabulary
  vocab = set(tokens)
  # Encode tokens numerically (replace with one-hot encoding or word embeddings)
  encoded_tokens = [vocab.index(token) for token in tokens]
  return encoded_tokens, vocab

encoded_tokens, vocab = preprocess_text(text)

# Prepare training data
def create_training_data(encoded_tokens, window_size=3):
  sequences = []
  labels = []
  for i in range(len(encoded_tokens) - window_size):
    sequence = encoded_tokens[i:i+window_size]
    label = encoded_tokens[i+window_size]
    sequences.append(sequence)
    labels.append(label)
  return np.array(sequences), np.array(labels)

x_train, y_train = create_training_data(encoded_tokens)

# Build the model
model = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=window_size),
  tf.keras.layers.LSTM(units),
  tf.keras.layers.Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(x_train, y_train, epochs=epochs)

# Generate text
def generate_text(model, start_text, max_length=100):
  generated_text = start_text
  for _ in range(max_length):
    # Encode the current text
    encoded_text = [vocab.index(word) for word in generated_text.split()[-window_size:]]
    # Predict the next word
    prediction = model.predict(np.array([encoded_text]))[0]
    predicted_word = vocab[np.argmax(prediction)]
    # Append the predicted word to the generated text
    generated_text += ' ' + predicted_word
  return generated_text

# Example usage
start_text = "The sun was shining brightly"
generated_text = generate_text(model, start_text)
print(generated_text)

