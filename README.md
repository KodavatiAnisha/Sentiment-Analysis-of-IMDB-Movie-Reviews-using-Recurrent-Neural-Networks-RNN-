# Sentiment-Analysis-of-IMDB-Movie-Reviews-using-Recurrent-Neural-Networks-RNN-
Built a sentiment analysis model using RNN on the IMDB movie reviews dataset. Preprocessed text data with embeddings and trained a Simple RNN to classify reviews as positive or negative. Evaluated performance using accuracy and visualized training progress.
Code Rundown for IMDB Sentiment Analysis using RNN
 1. Import Libraries
Import numpy, matplotlib, and deep learning modules from keras:
datasets, models, layers, preprocessing.

2. Load the IMDB Dataset
Load dataset using keras.datasets.imdb.

Keep only the top 10,000 most frequent words.

Reviews are already tokenized as sequences of integers.

3. Preprocess Data
Use pad_sequences() to pad or truncate reviews to the same length (e.g., 100 words).

Ensures uniform input size for the neural network.

4. Build the RNN Model
Use Sequential() model with the following layers:

Embedding(input_dim=10000, output_dim=32, input_length=100) – Converts word indices into dense vector representations.

SimpleRNN(units=32) – Learns sequence dependencies.

Dense(1, activation='sigmoid') – Outputs binary sentiment (positive or negative).

5. Compile the Model
Loss Function: 'binary_crossentropy'

Optimizer: 'adam'

Evaluation Metric: 'accuracy'

6. Train the Model
Train on 25,000 samples with validation_split=0.2.

Train for multiple epochs (e.g., 5) using a batch size (e.g., 128).

Model learns to predict sentiment based on word patterns.

 7. Evaluate the Model
Test the model on 25,000 unseen reviews.

Display test loss and test accuracy.

8. Visualize Performance
Plot training and validation accuracy/loss over epochs.

Helps identify overfitting, underfitting, and model learning trends.

