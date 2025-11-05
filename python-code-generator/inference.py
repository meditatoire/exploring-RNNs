import tensorflow as tf
from train import RNN, preprocess_data, setup_env

class InferenceModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="embedding")
        self.lstm_1 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, name="lstm_1")
        self.lstm_2 = tf.keras.layers.LSTM(rnn_units, return_sequences=True, return_state=True, name="lstm_2")
        self.dense = tf.keras.layers.Dense(vocab_size, name="dense")

    # manually define call method to handle states
    def call(self, inputs, states=None):
        if states is None:
            state_1 = self.lstm_1.get_initial_state(batch_size=tf.shape(inputs)[0])
            state_2 = self.lstm_2.get_initial_state(batch_size=tf.shape(inputs)[0])
        else:
            state_1 = states[0:2] # [h1, c1]
            state_2 = states[2:4] # [h2, c2]

        # Run the forward pass
        x = self.embedding(inputs)
        x, h1, c1 = self.lstm_1(x, initial_state=state_1)
        x, h2, c2 = self.lstm_2(x, initial_state=state_2)
        logits = self.dense(x)

        # Group the new states
        new_states = [h1, c1, h2, c2]
        return logits, new_states

def build_inference_model(trained_model):

    # Get the trained model configuration
    config = trained_model.get_config()
    vocab_size = config['vocab_size'] 
    embedding_dim = config['embedding_dim'] 
    rnn_units = config['rnn_units'] 

    inference_model = InferenceModel(vocab_size, embedding_dim, rnn_units)

    '''Passing a dummy input to build the model, this is due to architecture of the trained_model
    having an Input layer with a batch_shape model.build fail'''
    dummy_input = tf.constant([[0]]) 
    _ = inference_model(dummy_input) 

    trained_sequential_model = trained_model.model

    try:
        inference_model.embedding.set_weights(trained_sequential_model.get_layer('embedding').get_weights())
        inference_model.lstm_1.set_weights(trained_sequential_model.get_layer('lstm_1').get_weights())
        inference_model.lstm_2.set_weights(trained_sequential_model.get_layer('lstm_2').get_weights())
        inference_model.dense.set_weights(trained_sequential_model.get_layer('dense').get_weights())
    except Exception as e:
        print("Failed to set weights", e)

    return inference_model

def generate_text(inference_model, start_string, char_to_idx, idx_to_char, num_generate=1000, temperature=0.5):
    """
    Generates text using the stateful inference_model.
    """
    # Vectorize the start string
    input_eval = [char_to_idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0) # adding batch dimension shape: (1, seq_len)

    text_generated = []
    
    # WARM UP THE STATE
    # Run the start_string through the model to get its final hidden state
    logits, states = inference_model(input_eval, states=None)

    # Get the first predicted character
    last_step_logits = logits[:, -1, :] # Shape: (1, vocab_size)
    
    # Apply temperature and sampling
    last_step_logits = last_step_logits / temperature
    predicted_id = tf.random.categorical(last_step_logits, num_samples=1)[0, 0].numpy()

    # The next input is this first predicted character
    next_input = tf.expand_dims([predicted_id], 0) # adding batch dim shape: (1, 1)
    text_generated.append(idx_to_char[predicted_id])

    # Generation loop
    for i in range(num_generate - 1):
        # Run the model with the *previous states* and the *new character*
        logits, states = inference_model(
            next_input, 
            states=states
        )

        # The output logits are shape (1, 1, vocab_size)
        # Squeeze to (1, vocab_size) for sampling
        logits_squeezed = tf.squeeze(logits, axis=1)
        
        # Apply temperature and sampling
        logits_squeezed = logits_squeezed / temperature
        predicted_id = tf.random.categorical(logits_squeezed, num_samples=1)[0, 0].numpy()

        # Update the next_input and save the character
        next_input = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx_to_char[predicted_id])

    return (start_string + ''.join(text_generated))

def main():
    setup_env()

    #loading the character maping
    encoded_text, vocab, char_to_idx, idx_to_char, vocab_size = preprocess_data()
    
    trained_model = tf.keras.models.load_model('trained_model.keras')
    inference_model = build_inference_model(trained_model)

    try:
        num_generate = int(input("Enter number of characters to generate: "))
        temperature = float(input("Enter temperature: "))
        start_string = input("Enter start string: ")
        generated_text = generate_text(inference_model, start_string, char_to_idx, idx_to_char, num_generate, temperature)
        print(generated_text)
    except ValueError as e:
        print("Invalid input", e)
        print("generating with default values")
        generated_text = generate_text(inference_model, "import", char_to_idx, idx_to_char, 1000, 0.5)
        print(generated_text)


if __name__ == "__main__":
    main()
    



    