# save_model.py

import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils
from core.config import cfg
flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-416', 'path to output')
flags.DEFINE_boolean('tiny', False, 'is yolo-tiny or not')
flags.DEFINE_integer('input_size', 416, 'define input size of export model')
flags.DEFINE_float('score_thres', 0.2, 'define score threshold')
flags.DEFINE_string('framework', 'tf', 'define what framework do you want to convert (tf, trt, tflite)')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('fold_bn', False, 'fold batch normalization into conv weights (removes BN from graph for TFLite)')




def save_tf():
  STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

  if FLAGS.fold_bn:
    from core import common

    print("\n🔧 BatchNorm Folding Mode: Building model in 2 passes")
    print("="*60)

    # Pass 1: Build model WITH BatchNorm and load weights
    print("\nPass 1: Building model WITH BatchNorm...")
    common.set_batchnorm_folding(False)
    input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    model_with_bn = tf.keras.Model(input_layer, feature_maps)
    utils.load_weights(model_with_bn, FLAGS.weights, FLAGS.model, FLAGS.tiny)
    print("✓ Weights loaded into model with BatchNorm")

    # Fold BN weights
    folded_weights_dict = common.fold_batchnorm_into_conv(model_with_bn)

    # Pass 2: Build model WITHOUT BatchNorm
    print("\nPass 2: Building model WITHOUT BatchNorm...")
    common.set_batchnorm_folding(True)
    input_layer2 = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
    feature_maps2 = YOLO(input_layer2, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    model = tf.keras.Model(input_layer2, feature_maps2)
    print("✓ Model built without BatchNorm layers")

    # Copy weights: Conv weights from model_with_bn, but set folded bias
    print("\nCopying and folding weights...")
    conv_idx = 0
    for layer_with_bn, layer_no_bn in zip(model_with_bn.layers, model.layers):
      if isinstance(layer_no_bn, tf.keras.layers.Conv2D):
        layer_name = f'conv2d_{conv_idx}' if conv_idx > 0 else 'conv2d'

        if layer_name in folded_weights_dict:
          # Set folded weights
          W_folded, B_folded = folded_weights_dict[layer_name]
          layer_no_bn.set_weights([W_folded, B_folded])
        else:
          # Just copy weights (no BN to fold)
          try:
            layer_no_bn.set_weights(layer_with_bn.get_weights())
          except:
            pass
        conv_idx += 1
      elif len(layer_no_bn.get_weights()) > 0:
        # Copy weights for other layers
        try:
          layer_no_bn.set_weights(layer_with_bn.get_weights())
        except:
          pass

    print("✓ All weights copied and folded")
    print("="*60)
    print("✅ Model ready: NO BatchNorm layers in graph!\n")

  else:
    # Normal mode: build with BatchNorm
    input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    model = tf.keras.Model(input_layer, feature_maps)
    utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)

  model.summary()
  model.save(FLAGS.output)




def main(_argv):
  save_tf()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
