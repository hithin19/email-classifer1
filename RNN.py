import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import re
import string
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv('email.csv').sample(50000)

# Data cleaning
def clean_text(text):
    text = re.sub(r'\w*\d\w*', ' ', text)  # Remove words containing numbers
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text.lower())  # Lowercase and remove punctuation
    text = re.sub(r"\n", " ", text)  # Remove newlines
    text = re.sub(r'[^\x00-\x7f]', ' ', text)  # Remove non-ASCII characters
    return text

data['content'] = data['content'].apply(clean_text)

# Tokenize and pad sequences
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['content'].values)
X = tokenizer.texts_to_sequences(data['content'].values)
X = pad_sequences(X)

# Create target variable with KMeans clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)
data['Cluster'] = cluster_labels
y = data['Cluster']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the RNN model
embedding_dim = 128
model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))

# Compile the model with a lower learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print(model.summary())

# Train the model
epochs = 25
batch_size = 32
early_stopping = EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                    validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

# Save the model and tokenizer
model.save('rnn_model.h5')
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Evaluate the model
def evaluate_rnn(model, X_train, X_test, y_train, y_test):
    y_train_pred = np.argmax(model.predict(X_train), axis=-1)
    y_test_pred = np.argmax(model.predict(X_test), axis=-1)

    print("TRAINING RESULTS: \n===============================")
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{classification_report(y_train, y_train_pred)}")

    print("\nTESTING RESULTS: \n===============================")
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{classification_report(y_test, y_test_pred)}")

evaluate_rnn(model, X_train, X_test, y_train, y_test)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
