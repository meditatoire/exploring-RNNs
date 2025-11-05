import tensorflow as tf
import string
import numpy as np
import matplotlib.pyplot as plt

def setup_env():
    #Using float16 for training
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    #forcing tf to use 3.5 gb in vram of my 4gb vram gpu (to avoid OOM)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=3500)]
            )

def preprocess_data():
    ds_text = ''
    with open('data.txt', "r" ) as f:
        for line in f:
            ds_text += line.rstrip() + '\n' #removing trailing whitespace
    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + ' \t\n')
    ds_text = ''.join(ch for ch in ds_text if ch in allowed_chars)

    #character level tokenization
    vocab = sorted(set(ds_text))
    char_to_idx = {char:i for i,char in enumerate(vocab)}
    idx_to_char = {i:char for char, i in char_to_idx.items()}
    encoded_text = np.array([char_to_idx[c] for c in ds_text])
    return encoded_text, vocab, char_to_idx, idx_to_char, len(vocab)

#creating dataset
def create_tf_dataset(encoded_text, seq_length, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
    sequences = dataset.batch(seq_length+1, drop_remainder=True)
    
    def split_input_target(sequence):
        input_text = sequence[:-1]
        target_text = sequence[1:]
        return input_text, target_text
    
    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
    return dataset

#building model
@tf.keras.utils.register_keras_serializable()
class RNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.batch_size = batch_size

        self.model = tf.keras.models.Sequential([
            tf.keras.Input(batch_shape=[batch_size, None]),
            tf.keras.layers.Embedding(vocab_size, embedding_dim, name="embedding"),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=False, name="lstm_1"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=False, name="lstm_2"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(vocab_size, kernel_regularizer=tf.keras.regularizers.l2(1e-4), name="dense")
        ])
    def call(self, x):
        return self.model(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'rnn_units': self.rnn_units,
            'batch_size': self.batch_size,
        })
        return config

def plot_history(history):
    plt.figure(figsize=(18, 7))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def main():
    #setup environment
    setup_env()

    #hyperparameters
    seq_length = 256
    batch_size = 512
    embedding_dim = 64
    rnn_units = 256

    #preprocessing data
    print("Preprocessing data...")
    encoded_text, vocab, char_to_idx, idx_to_char, vocab_size = preprocess_data()
    
    #train - test split
    print("Splitting data...")
    train_size = int(0.95 * len(encoded_text))
    train_data = encoded_text[:train_size]
    test_data = encoded_text[train_size:]
    

    #creating datasets
    print("Creating datasets...")
    train_dataset = create_tf_dataset(train_data, seq_length, batch_size)
    test_dataset = create_tf_dataset(test_data, seq_length, batch_size)

    #building model
    print("Building model...")
    model = RNN(vocab_size, embedding_dim, rnn_units, batch_size)
    model.model.summary()
    
    #compiling model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1.8e-4, clipnorm=1),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])
    
    #setting up checkpoint callback to save best model
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="model_checkpoint.keras",  
        save_weights_only=False,
        save_best_only=True,
        save_freq='epoch' ,
        monitor='val_loss',
        mode = 'min'
    )
    
    #training model
    print("Training model...")
    history = model.fit(train_dataset, epochs=300, validation_data=test_dataset, callbacks=[checkpoint_callback])
    print("Training completed. Best model saved to model_checkpoint.keras")
    
    #plotting history
    print("Plotting history...")
    plot_history(history)
    
if __name__ == "__main__":
    main()
    

    

