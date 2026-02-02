"""
Quick Start: Structured Channel Pruning for YOLOv4-Tiny

This is a simplified version that demonstrates structured pruning.
Install requirements: pip install kerassurgeon

Usage:
    python quick_structured_prune.py --target_layer conv2d_17 --prune_ratio 0.3
"""

import tensorflow as tf
import numpy as np
from absl import app, flags, logging
from absl.flags import FLAGS
import os

from core.yolov4 import YOLO
import core.utils as utils

# Flags
flags.DEFINE_string('weights', './data/yolov4-tiny.weights', 'weights path')
flags.DEFINE_string('output', './checkpoints/yolov4-tiny-pruned', 'output path')
flags.DEFINE_string('target_layer', 'conv2d_17', 'layer to prune')
flags.DEFINE_float('prune_ratio', 0.3, 'prune ratio (0.3 = 30%)')
flags.DEFINE_integer('input_size', 416, 'input size')
flags.DEFINE_boolean('tiny', True, 'is tiny')
flags.DEFINE_string('model', 'yolov4', 'model type')

def fpgm_score(weights):
    """Calculate FPGM scores for filters"""
    num_filters = weights.shape[-1]
    filters = weights.reshape(-1, num_filters)

    scores = []
    for i in range(num_filters):
        f = filters[:, i:i+1]
        dist = np.linalg.norm(filters - f, axis=0)
        scores.append(np.sum(dist))

    return np.array(scores)

def get_channels_to_prune(layer, ratio):
    """Get list of channel indices to prune"""
    weights = layer.get_weights()[0]
    scores = fpgm_score(weights)

    num_to_prune = int(len(scores) * ratio)
    return np.argsort(scores)[:num_to_prune].tolist()

def main(_argv):
    logging.info("="*60)
    logging.info("STRUCTURED CHANNEL PRUNING")
    logging.info("="*60)

    # Check keras-surgeon
    try:
        from kerassurgeon import Surgeon
    except ImportError:
        logging.error("keras-surgeon not found!")
        logging.error("Install: pip install kerassurgeon")
        return

    # Load model
    logging.info(f"\nLoading model from {FLAGS.weights}...")
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    model = tf.keras.Model(input_layer, feature_maps)

    if FLAGS.weights.endswith('.weights'):
        utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
    else:
        model.load_weights(FLAGS.weights)

    original_params = model.count_params()
    logging.info(f"Original parameters: {original_params:,}")

    # Get layer
    try:
        layer = model.get_layer(FLAGS.target_layer)
    except:
        logging.error(f"Layer '{FLAGS.target_layer}' not found!")
        logging.info("\nAvailable Conv2D layers:")
        for l in model.layers:
            if isinstance(l, tf.keras.layers.Conv2D):
                logging.info(f"  {l.name} - {l.filters} filters")
        return

    logging.info(f"\nPruning layer: {FLAGS.target_layer}")
    logging.info(f"  Original filters: {layer.filters}")

    # Calculate channels to prune
    channels = get_channels_to_prune(layer, FLAGS.prune_ratio)
    num_pruned = len(channels)
    num_remaining = layer.filters - num_pruned

    logging.info(f"  Pruning: {num_pruned} filters ({FLAGS.prune_ratio*100:.0f}%)")
    logging.info(f"  Remaining: {num_remaining} filters")
    logging.info(f"  Channels: {channels[:5]}..." if len(channels) > 5 else f"  Channels: {channels}")

    # Prune with Surgeon
    logging.info("\nApplying structured pruning...")
    surgeon = Surgeon(model, copy=True)
    surgeon.add_job('delete_channels', layer, channels=channels)
    pruned_model = surgeon.operate()

    pruned_params = pruned_model.count_params()
    reduction = (original_params - pruned_params) / original_params * 100

    logging.info(f"\n✓ Pruning complete!")
    logging.info(f"  Pruned parameters: {pruned_params:,}")
    logging.info(f"  Reduction: {reduction:.1f}%")

    # Save
    logging.info(f"\nSaving to {FLAGS.output}...")
    os.makedirs(os.path.dirname(FLAGS.output) if os.path.dirname(FLAGS.output) else '.', exist_ok=True)

    pruned_model.save_weights(FLAGS.output + '.h5')
    tf.saved_model.save(pruned_model, FLAGS.output + '_savedmodel')

    logging.info("="*60)
    logging.info("DONE!")
    logging.info(f"Output: {FLAGS.output}.h5")
    logging.info(f"        {FLAGS.output}_savedmodel/")
    logging.info("\nNext:")
    logging.info("  1. Test with detect.py")
    logging.info("  2. Evaluate with evaluate.py")
    logging.info("  3. Fine-tune with structured_prune.py if accuracy drops")
    logging.info("     (Use --train_dataset to enable fine-tuning)")
    logging.info("="*60)

if __name__ == '__main__':
    app.run(main)
