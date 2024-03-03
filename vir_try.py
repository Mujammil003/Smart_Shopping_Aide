import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as pltpy

# Load your dataset (assuming it's in CSV format)
dataset_path = 'Clothing.csv'
df = pd.read_csv(dataset_path)

df = df.sample(frac=0.1, random_state=42)  # Sample 10% of the dataset
# Combine relevant features into a single description
df['combined_features'] = df['name'] + ' ' + df['main_category'] + ' ' + df['sub_category'] + ' ' + df['ratings'].astype(str) + ' stars ' + df['discount_price'].astype(str) + ' off ' + df['actual_price'].astype(str)

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['combined_features'])
total_words = len(tokenizer.word_index) + 1

# Create input sequences
input_sequences = []
for line in df['combined_features']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Create predictors and labels
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and compile the model
model = Sequential()
model.add(Embedding(total_words, 150, input_length=max_sequence_length-1))
model.add(LSTM(150, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(100, dropout=0.2))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
st.write(f'Test Loss: {loss:.4f}')
st.write(f'Test Accuracy: {accuracy * 100:.2f}%')

# Function to generate product description
def generate_product_description(seed_text, next_words, model, max_sequence_length, tokenizer):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')

        # Use predict to get probabilities for each word
        predicted_probabilities = model.predict(token_list, verbose=0)[0]

        # Select the word with the highest probability
        predicted_word_index = tf.argmax(predicted_probabilities).numpy()
        predicted_word = tokenizer.index_word[predicted_word_index]
        seed_text += " " + predicted_word
    return seed_text

# Streamlit UI
st.title("Product Description Generator")

# User input for new product details
new_product_input = st.text_input("Enter the new product details:", "")

# Generate product description on button click
if st.button("Generate Description"):
    generated_description = generate_product_description(new_product_input, 10, model, max_sequence_length, tokenizer)
    st.subheader("Generated Product Description:")
    st.write(generated_description)

# Plot training & validation accuracy values
st.subheader("Model Accuracy Over Epochs")
fig_acc, ax_acc = plt.subplots()
ax_acc.plot(history.history['accuracy'], label='Train')
ax_acc.plot(history.history['val_accuracy'], label='Validation')
ax_acc.set_xlabel('Epoch')
ax_acc.set_ylabel('Accuracy')
ax_acc.legend()
st.pyplot(fig_acc)

# Plot training & validation loss values
st.subheader("Model Loss Over Epochs")
fig_loss, ax_loss = plt.subplots()
ax_loss.plot(history.history['loss'], label='Train')
ax_loss.plot(history.history['val_loss'], label='Validation')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.legend()
st.pyplot(fig_loss)