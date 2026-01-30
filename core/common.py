#! /usr/bin/env python
# coding=utf-8

import tensorflow as tf
import numpy as np

# Global flag to control BatchNorm folding
# When True, BN layers are NOT added to the graph (weights are folded into Conv)
_FOLD_BATCHNORM = False
_BN_WEIGHTS_CACHE = {}

def set_batchnorm_folding(enabled):
    """Enable or disable batchnorm folding. When enabled, BN layers won't be added to the graph."""
    global _FOLD_BATCHNORM
    _FOLD_BATCHNORM = enabled

def cache_bn_weights(conv_name, bn_weights):
    """Cache BN weights for a conv layer to be folded later."""
    global _BN_WEIGHTS_CACHE
    _BN_WEIGHTS_CACHE[conv_name] = bn_weights

def get_cached_bn_weights(conv_name):
    """Get cached BN weights for a conv layer."""
    return _BN_WEIGHTS_CACHE.get(conv_name, None)

# import tensorflow_addons as tfa
class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    "Frozen state" and "inference mode" are two separate concepts.
    `layer.trainable = False` is to freeze the layer, so the layer will use
    stored moving `var` and `mean` in the "inference mode", and both `gama`
    and `beta` will not be updated !
    """

    def call(self, x, training=False):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)

def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky'):
    global _FOLD_BATCHNORM

    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
        padding = 'valid'
        strides = 2
    else:
        strides = 1
        padding = 'same'

    # If folding is enabled and BN is requested, use bias directly (will be set to folded BN weights later)
    use_bias = (not bn) or _FOLD_BATCHNORM

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size = filters_shape[0], strides=strides, padding=padding,
                                  use_bias=use_bias, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.constant_initializer(0.))(input_layer)

    # Only add BN layer if NOT folding
    if bn and not _FOLD_BATCHNORM:
        conv = BatchNormalization()(conv)

    if activate == True:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
    return conv

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))
    # return tf.keras.layers.Lambda(lambda x: x*tf.tanh(tf.math.log(1+tf.exp(x))))(x)

def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky'):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type)
    conv = convolutional(conv       , filters_shape=(3, 3, filter_num1,   filter_num2), activate_type=activate_type)

    residual_output = short_cut + conv
    return residual_output

# def block_tiny(input_layer, input_channel, filter_num1, activate_type='leaky'):
#     conv = convolutional(input_layer, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#     short_cut = input_layer
#     conv = convolutional(conv, filters_shape=(3, 3, input_channel, filter_num1), activate_type=activate_type)
#
#     input_data = tf.concat([conv, short_cut], axis=-1)
#     return residual_output

def route_group(input_layer, groups, group_id):
    convs = tf.split(input_layer, num_or_size_splits=groups, axis=-1)
    return convs[group_id]

def upsample(input_layer):
    return tf.image.resize(input_layer, (input_layer.shape[1] * 2, input_layer.shape[2] * 2), method='bilinear')

def fold_batchnorm_into_conv(model):
    """
    Fold BatchNorm weights from a model WITH BN into a model built WITHOUT BN.
    This loads the folded weights into Conv bias terms.

    Args:
        model: Model built WITH _FOLD_BATCHNORM=False (has BN layers)

    Returns:
        Model built WITH _FOLD_BATCHNORM=True (no BN layers, folded weights)
    """
    print("\n" + "="*60)
    print("FOLDING BATCH NORMALIZATION INTO CONV WEIGHTS")
    print("="*60)

    # Extract folded weights from the model with BN
    folded_weights = {}
    conv_idx = 0

    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Look for BN after this conv
            for j in range(i + 1, min(i + 3, len(model.layers))):
                next_layer = model.layers[j]
                if isinstance(next_layer, (tf.keras.layers.BatchNormalization, BatchNormalization)):
                    # Found Conv->BN pair, fold it
                    conv_weights = layer.get_weights()
                    bn_weights = next_layer.get_weights()

                    if len(conv_weights) == 1:  # Conv without bias
                        W = conv_weights[0]
                        gamma, beta, moving_mean, moving_variance = bn_weights
                        epsilon = next_layer.epsilon

                        # Compute folded weights
                        scale = gamma / np.sqrt(moving_variance + epsilon)
                        out_channels = W.shape[-1]

                        if len(scale) == out_channels:
                            W_folded = W * scale.reshape(1, 1, 1, -1)
                            B_folded = beta - gamma * moving_mean / np.sqrt(moving_variance + epsilon)
                            folded_weights[f'conv2d_{conv_idx}'] = (W_folded, B_folded)
                            print(f"✓ Folded: {layer.name:35s} + {next_layer.name}")
                        else:
                            print(f"⚠ Skipped {layer.name}: dimension mismatch")
                    elif len(conv_weights) == 2:  # Conv with existing bias
                        W, B = conv_weights
                        gamma, beta, moving_mean, moving_variance = bn_weights
                        epsilon = next_layer.epsilon

                        scale = gamma / np.sqrt(moving_variance + epsilon)
                        out_channels = W.shape[-1]

                        if len(scale) == out_channels:
                            W_folded = W * scale.reshape(1, 1, 1, -1)
                            B_folded = beta + (B - moving_mean) * scale
                            folded_weights[f'conv2d_{conv_idx}'] = (W_folded, B_folded)
                            print(f"✓ Folded: {layer.name:35s} + {next_layer.name}")
                        else:
                            print(f"⚠ Skipped {layer.name}: dimension mismatch")
                    break
            conv_idx += 1

    print(f"\nTotal folded: {len(folded_weights)} Conv->BN pairs")
    print("="*60 + "\n")

    return folded_weights
