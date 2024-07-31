import tensorflow as tf

class HybridAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, local_window_size, **kwargs):
        super(HybridAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.local_window_size = local_window_size
        
        # Global multi-head attention
        self.global_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=0.1
        )
        
        # Local multi-head attention
        self.local_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=0.1
        )
        
    def build(self, input_shape):
        super(HybridAttention, self).build(input_shape)

    def call(self, inputs):
        # Extract inputs
        query, key, value = inputs
        
        # Global attention
        global_output = self.global_attention(query, key, value)
        
        # Local attention
        seq_len = tf.shape(query)[1]
        
        # TensorArray to collect local attention results
        local_outputs = tf.TensorArray(dtype=tf.float32, size=seq_len, dynamic_size=False)
        
        for i in tf.range(seq_len):
            # Define local window using TensorFlow operations
            start = tf.maximum(0, i - self.local_window_size // 2)
            end = tf.minimum(seq_len, i + self.local_window_size // 2 + 1)
            
            # Slice for local window
            local_query = query[:, start:end, :]
            local_key = key[:, start:end, :]
            local_value = value[:, start:end, :]
            
            # Apply local attention
            local_output = self.local_attention(local_query, local_key, local_value)
            
            # Write to TensorArray
            local_outputs = local_outputs.write(i, local_output)
        
        # Concatenate local attention results
        local_output = local_outputs.stack()
        
        # Combine global and local outputs
        combined_output = global_output + tf.reduce_sum(local_output, axis=1)
        
        return combined_output

# Example usage
input_shape = (None, 50, 64)  # (batch_size, sequence_length, embedding_dim)
inputs = tf.keras.Input(shape=input_shape[1:])
hybrid_attention_layer = HybridAttention(num_heads=4, d_model=64, local_window_size=5)
outputs = hybrid_attention_layer([inputs, inputs, inputs])
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Print model summary
model.summary()
