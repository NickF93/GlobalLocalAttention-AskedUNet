import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import plot_model

class UNetWithAttention:
    """
    A U-Net architecture with attention mechanisms and configurable skip connections.

    Attributes:
        input_shape (tuple): The shape of the input image, e.g., (128, 128, 1) for grayscale images.
        timestamp_dim (int): The dimension of the timestamp input.
        filter_list (list): A list of integers representing the number of filters for each encoder/decoder block.
        num_skip_connections (int): The number of skip connections from the deepest layer.
        num_heads (int): The number of attention heads in the multi-head attention layers.
        key_dim (int): The dimensionality of the key space for the multi-head attention.
        use_bias (bool): Whether to include biases in the layers.
        activation (str): The activation function to use in the convolutional layers.
        model (Model): The Keras model instance.
    """

    def __init__(self, input_shape, timestamp_dim, filter_list, num_skip_connections, num_heads=4, key_dim=64, use_bias=False, activation='swish'):
        """
        Initializes the UNetWithAttention class.

        Args:
            input_shape (tuple): The shape of the input images.
            timestamp_dim (int): The dimensionality of the timestamp input.
            filter_list (list): A list of filters for each encoder and decoder block.
            num_skip_connections (int): The number of skip connections to include, starting from the deepest layer.
            num_heads (int): The number of heads for the multi-head attention layers.
            key_dim (int): The dimensionality of the key vectors for multi-head attention.
            use_bias (bool): Whether to use biases in convolutional layers and attention mechanisms.
            activation (str): The activation function to use in the convolutional layers.
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
        Creates a convolutional block consisting of Conv2D, BatchNormalization, and the specified activation.

        Args:
            x (tensor): Input tensor.
            filters (int): Number of filters for the convolutional layers.

        Returns:
            tensor: Output tensor after applying the block.
        """
        x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=self.use_bias)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(self.activation)(x)
        return x

    def _residual_block(self, x, filters):
        """
        Creates a residual block with a skip connection.

        Args:
            x (tensor): Input tensor.
            filters (int): Number of filters for the convolutional layers.

        Returns:
            tensor: Output tensor after applying the block.
        """
        res = layers.Conv2D(filters, (1, 1), padding='same', use_bias=self.use_bias)(x)
        x = self._conv_block(x, filters)
        x = self._conv_block(x, filters)
        x = layers.Add()([x, res])
        x = layers.Activation(self.activation)(x)
        return x

    def _multihead_attention_block(self, x):
        """
        Applies a multi-head attention mechanism followed by a residual connection and layer normalization.

        Args:
            x (tensor): Input tensor.

        Returns:
            tensor: Output tensor after applying the attention block.
        """
        attn_output = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.key_dim, use_bias=self.use_bias)(x, x)
        attn_output = layers.Add()([x, attn_output])
        attn_output = layers.LayerNormalization()(attn_output)
        return attn_output

    def _positional_embedding(self, x):
        """
        Adds positional embeddings to the input tensor.

        Args:
            x (tensor): Input tensor.

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
            x (tensor): Input tensor.
            filters (int): Number of filters for the convolutional layers.

        Returns:
            tensor, tensor: Output tensor and downsampled tensor.
        """
        x = self._residual_block(x, filters)
        p = layers.Conv2D(filters, (3, 3), strides=(2, 2), padding='same', use_bias=self.use_bias)(x)
        return x, p

    def _decoder_block(self, x, skip_features, filters):
        """
        Creates a decoder block consisting of upsampling, concatenation with skip connections, and a residual block.

        Args:
            x (tensor): Input tensor.
            skip_features (tensor): Tensor from the skip connections.
            filters (int): Number of filters for the convolutional layers.

        Returns:
            tensor: Output tensor after applying the block.
        """
        x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same', use_bias=self.use_bias)(x)
        x = layers.Concatenate()([x, skip_features])
        x = self._positional_embedding(x)
        x = self._multihead_attention_block(x)
        x = self._residual_block(x, filters)
        return x

    def build_model(self):
        """
        Builds the U-Net model with attention mechanisms and stores it in the `model` attribute.
        """
        # Image input
        img_input = Input(self.input_shape)
        
        # Timestamp input
        time_input = Input((self.timestamp_dim,))

        # Encoder
        skip_connections = []
        x = img_input
        for filters in self.filter_list:
            x, x_down = self._encoder_block(x, filters)
            skip_connections.append(x)
            x = x_down

        # Bottleneck with multi-head attention
        b1 = self._residual_block(x, self.filter_list[-1])
        b1 = self._positional_embedding(b1)
        b1 = self._multihead_attention_block(b1)

        # Incorporating timestamps into attention (optional)
        time_expanded = layers.Dense(self.key_dim, use_bias=self.use_bias)(time_input)
        time_expanded = layers.Reshape((1, 1, self.key_dim))(time_expanded)
        time_expanded = layers.UpSampling2D(size=(b1.shape[1], b1.shape[2]))(time_expanded)
        b1 = layers.Concatenate()([b1, time_expanded])

        # Decoder with configurable skip connections
        skip_connections = skip_connections[::-1]  # Reverse to start from the deepest layer
        selected_skip_connections = skip_connections[:self.num_skip_connections]
        for filters, skip_features in zip(self.filter_list[::-1], selected_skip_connections):
            b1 = self._decoder_block(b1, skip_features, filters)

        # Output
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
            plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)
        else:
            print("Model has not been built yet. Call `build_model()` first.")

# Example usage
input_shape = (128, 128, 1)  # Example input shape for grayscale images
timestamp_dim = 1  # Example dimension for timestamp (can be a single value or a vector)
filter_list = [64, 128, 256, 512]  # List of filters for each encoder/decoder block
num_skip_connections = 3  # Number of skip connections from the deepest layer

unet_model = UNetWithAttention(input_shape, timestamp_dim, filter_list, num_skip_connections, use_bias=False)
unet_model.build_model()
unet_model.print_model()
unet_model.save_model_plot('unet_with_attention.png')
