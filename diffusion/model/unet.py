import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import plot_model

class UNetWithAttention:
    """
    A U-Net architecture enhanced with attention mechanisms and configurable skip connections.

    Attributes:
        input_shape (tuple): The shape of the input image tensor, e.g., (224, 224, 3) for RGB images.
        timestamp_dim (int): The dimension of the timestamp input, e.g., 1 for a scalar timestamp.
        filter_list (list): A list of integers representing the number of filters for each encoder and decoder block.
        num_skip_connections (int): The number of skip connections to include from the deepest layer of the encoder.
        num_heads (int): The number of attention heads in the multi-head attention layers.
        key_dim (int): The dimensionality of the key vectors for the multi-head attention layers.
        use_bias (bool): Whether to include biases in convolutional layers and attention mechanisms.
        activation (str): The activation function to use in the convolutional layers, e.g., 'swish'.
        model (Model): The Keras model instance representing the U-Net with attention architecture.
    """

    def __init__(self, input_shape, timestamp_dim, filter_list, num_skip_connections, num_heads=4, key_dim=64, use_bias=False, activation='swish'):
        """
        Initializes the UNetWithAttention class with the specified parameters.

        Args:
            input_shape (tuple): The shape of the input images, e.g., (224, 224, 3).
            timestamp_dim (int): The dimensionality of the timestamp input.
            filter_list (list): A list of integers where each integer represents the number of filters for the corresponding encoder/decoder block.
            num_skip_connections (int): Number of skip connections from the deepest layer of the encoder.
            num_heads (int): Number of attention heads for the multi-head attention layers.
            key_dim (int): Dimensionality of the key vectors for the multi-head attention mechanism.
            use_bias (bool): Whether to use biases in convolutional layers and attention mechanisms.
            activation (str): The activation function to use, e.g., 'swish'.
        """
        self.input_shape = input_shape
        self.timestamp_dim = timestamp_dim
        self.filter_list = filter_list
        self.num_skip_connections = num_skip_connections
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.use_bias = use_bias
        self.activation = activation
        self.model = None

    def _conv_block(self, x, filters):
        """
        Creates a convolutional block consisting of Conv2D, BatchNormalization, and the specified activation function.

        Args:
            x (tensor): Input tensor to the convolutional block.
            filters (int): Number of filters for the Conv2D layers.

        Returns:
            tensor: Output tensor after applying the convolutional block.
        """
        x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=self.use_bias)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)
        return x

    def _residual_block(self, x, filters):
        """
        Creates a residual block with a skip connection, which consists of two convolutional blocks and a residual connection.

        Args:
            x (tensor): Input tensor to the residual block.
            filters (int): Number of filters for the convolutional layers.

        Returns:
            tensor: Output tensor after applying the residual block.
        """
        res = layers.Conv2D(filters, (1, 1), padding='same', use_bias=self.use_bias)(x)
        x = self._conv_block(x, filters)
        x = self._conv_block(x, filters)
        x = layers.Add()([x, res])
        x = layers.Activation(self.activation)(x)
        return x

    def _neighborhood_attention(self, query, key, value, num_heads, key_dim, neighborhood_size, dropout_rate=0.1):
        """
        Applies Neighborhood Attention mechanism using MultiHeadAttention.

        Args:
            query (tensor): Input tensor for query (shape: [batch_size, height, width, depth]).
            key (tensor): Input tensor for key (shape: [batch_size, height, width, depth]).
            value (tensor): Input tensor for value (shape: [batch_size, height, width, depth]).
            num_heads (int): Number of attention heads.
            key_dim (int): Dimensionality of the key vectors.
            neighborhood_size (int): Number of neighboring tokens each token can attend to.
            dropout_rate (float): Dropout rate for regularization.

        Returns:
            tensor: Output tensor after applying neighborhood attention.
        """
        mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)
        
        height, width = tf.shape(query)[1:3]
        seq_len = height * width
        depth = tf.shape(query)[-1]
        
        query = tf.reshape(query, (-1, seq_len, depth))
        key = tf.reshape(key, (-1, seq_len, depth))
        value = tf.reshape(value, (-1, seq_len, depth)) if value is not None else query
        
        def create_neighborhood_mask(seq_len, neighborhood_size):
            mask = tf.ones((seq_len, seq_len), dtype=tf.float32)
            indices = tf.range(seq_len)
            indices = tf.expand_dims(indices, 0)
            distances = tf.abs(indices - tf.transpose(indices))
            mask = tf.cast(distances <= neighborhood_size, dtype=tf.float32)
            return mask
        
        mask = create_neighborhood_mask(seq_len, neighborhood_size)
        mask = tf.expand_dims(mask, 0)
        
        attention_output = mha(query, key, value, attention_mask=mask)
        
        batch_size = tf.shape(query)[0]
        attention_output = tf.reshape(attention_output, (batch_size, height, width, depth))

        return attention_output

    def _multihead_attention_block(self, x):
        """
        Applies a multi-head attention mechanism followed by a residual connection, layer normalization, and activation.

        Args:
            x (tensor): Input tensor to the multi-head attention block.

        Returns:
            tensor: Output tensor after applying the multi-head attention block.
        """
        attn_output = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, use_bias=self.use_bias, dropout=0.0)(x, x)
        nttn_output = self._neighborhood_attention(query=x, key=x, value=None, num_heads=self.num_heads, key_dim=self.key_dim, neighborhood_size=4, dropout_rate=0.0)
        attn_output = layers.Add()([x, attn_output, nttn_output])
        attn_output = layers.LayerNormalization()(attn_output)
        attn_output = layers.Activation(self.activation)(attn_output)
        return attn_output

    def _positional_embedding(self, x):
        """
        Adds positional embeddings to the input tensor to incorporate positional information.

        Args:
            x (tensor): Input tensor to which positional embeddings are added.

        Returns:
            tensor: Output tensor with positional embeddings added.
        """
        _, height, width, channels = x.shape
        num_positions = int(height) * int(width)
        positions = tf.range(start=0, limit=num_positions, delta=1)
        positions = tf.reshape(positions, (1, num_positions))
        pos_emb = tf.keras.layers.Embedding(input_dim=num_positions, output_dim=int(channels))(positions)
        pos_emb = tf.reshape(pos_emb, (1, int(height), int(width), int(channels)))
        return x + pos_emb

    def _encoder_block(self, x, filters):
        """
        Creates an encoder block consisting of a residual block followed by a downsampling operation.

        Args:
            x (tensor): Input tensor to the encoder block.
            filters (int): Number of filters for the convolutional layers.

        Returns:
            tensor, tensor: Output tensor after applying the residual block and the downsampled tensor.
        """
        x = self._residual_block(x, filters)
        p = layers.Conv2D(filters, (3, 3), strides=(2, 2), padding='same', use_bias=self.use_bias)(x)
        return x, p

    def _decoder_block(self, x, skip_features, filters):
        """
        Creates a decoder block consisting of upsampling, concatenation with skip connections, and a residual block.

        Args:
            x (tensor): Input tensor to the decoder block.
            skip_features (tensor): Tensor from the skip connections to concatenate with.
            filters (int): Number of filters for the convolutional layers.

        Returns:
            tensor: Output tensor after applying the decoder block.
        """
        x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', use_bias=self.use_bias)(x)
        x = layers.Concatenate()([x, skip_features])
        x = self._positional_embedding(x)
        x = self._multihead_attention_block(x)
        x = self._residual_block(x, filters)
        return x

    def build_model(self):
        """
        Constructs the U-Net model with attention mechanisms and stores it in the `model` attribute.
        """
        img_input = Input(self.input_shape)
        time_input = Input((self.timestamp_dim,))

        skip_connections = []
        x = img_input
        for filters in self.filter_list:
            x, x_down = self._encoder_block(x, filters)
            skip_connections.append(x)
            x = x_down

        b1 = self._residual_block(x, self.filter_list[-1])
        b1 = self._positional_embedding(b1)
        b1 = self._multihead_attention_block(b1)

        time_expanded = layers.Dense(self.key_dim, use_bias=self.use_bias)(time_input)
        time_expanded = layers.Reshape((1, 1, self.key_dim))(time_expanded)
        time_expanded = layers.UpSampling2D(size=(b1.shape[1], b1.shape[2]))(time_expanded)
        b1 = layers.Concatenate()([b1, time_expanded])

        skip_connections = skip_connections[::-1]
        selected_skip_connections = skip_connections[:self.num_skip_connections]
        for filters, skip_features in zip(self.filter_list[::-1], selected_skip_connections):
            b1 = self._decoder_block(b1, skip_features, filters)

        outputs = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid', use_bias=self.use_bias)(b1)

        self.model = Model([img_input, time_input], outputs)

    def print_model(self):
        """
        Prints the summary of the built model.
        """
        if self.model is not None:
            self.model.summary()
        else:
            print("Model has not been built yet. Call `build_model()` first.")

    def save_model_plot(self, filename='unet_model.png'):
        """
        Saves a plot of the model architecture to a file.

        Args:
            filename (str): The filename for the plot image.
        """
        if self.model is not None:
            plot_model(self.model, to_file=filename, show_shapes=True, show_dtype=True, show_layer_names=True, expand_nested=True, show_layer_activations=True, show_trainable=True)
        else:
            print("Model has not been built yet. Call `build_model()` first.")

# Test the model with random inputs
def test_model():
    """
    Tests the UNetWithAttention model with random data for different batch sizes.
    """
    input_shape = (224, 224, 3)  # Example input shape for RGB images
    timestamp_dim = 1  # Example dimension for timestamp (scalar value)
    filter_list = [32, 64, 128, 256, 512]  # Example filter list
    num_skip_connections = 3  # Example number of skip connections

    # Initialize the model
    unet_model = UNetWithAttention(input_shape, timestamp_dim, filter_list, num_skip_connections, use_bias=False)
    print('Build model')
    unet_model.build_model()
    print('Print model')
    unet_model.print_model()
    print('Plot model')
    unet_model.save_model_plot(filename='/tmp/unet.png')

    print('Test model')
    # Create random test data
    batch_sizes = [16, 1]
    for batch_size in batch_sizes:
        img_data = np.random.random((batch_size, *input_shape)).astype(np.float32)
        timestamp_data = np.random.random((batch_size, timestamp_dim)).astype(np.float32)
        
        # Perform prediction
        predictions = unet_model.model.predict([img_data, timestamp_data])
        print(f"Batch size: {batch_size}, Prediction shape: {predictions.shape}")

# Run the test
test_model()
