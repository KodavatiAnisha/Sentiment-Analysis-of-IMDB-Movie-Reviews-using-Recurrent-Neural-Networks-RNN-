# Sentiment-Analysis-of-IMDB-Movie-Reviews-using-Recurrent-Neural-Networks-RNN-
Built a sentiment analysis model using RNN on the IMDB movie reviews dataset. Preprocessed text data with embeddings and trained a Simple RNN to classify reviews as positive or negative. Evaluated performance using accuracy and visualized training progress.
Code Rundown for IMDB Sentiment Analysis using RNN
Import Libraries

Imports essential libraries like numpy, matplotlib, and deep learning modules from keras:
datasets, models, layers, preprocessing.

Load the IMDB Dataset

Loads the IMDB dataset using keras.datasets.imdb.

Limits vocabulary to the top 10,000 most frequent words.

Reviews are already tokenized as sequences of integers.

Preprocess Data

Uses pad_sequences() to make all reviews the same length (e.g., 100 words).

Ensures uniform input shape for the model.

Build the RNN Model

Uses Sequential() to stack layers:

Embedding(input_dim=10000, output_dim=32, input_length=100) – converts words to dense vectors.

SimpleRNN(units=32) – captures sequence patterns in the reviews.

Dense(1, activation='sigmoid') – outputs binary sentiment (0 = negative, 1 = positive).

Compile the Model

Loss function: 'binary_crossentropy'

Optimizer: 'adam'

Metrics: 'accuracy'

Train the Model

Trained on 25,000 reviews.

Uses validation_split=0.2 to monitor overfitting.

Trained for multiple epochs (e.g., 5) with a batch size (e.g., 128).

Evaluate the Model

Evaluated on the test set of 25,000 reviews.

Outputs test loss and accuracy.

Visualize Performance

Plots training and validation accuracy/loss over epochs.

Helps in understanding model learning behavior and overfitting.
