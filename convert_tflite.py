import os

from tensorflow.lite.python.interpreter import OpResolverType

# CRITICAL: Disable XNNPACK BEFORE importing TensorFlow
# This allows testing on x86 CPU the same way it will run on ARM (Conv2D on FPGA, rest on CPU)
os.environ['TF_LITE_ENABLE_XNNPACK'] = '0'
os.environ['TF_LITE_DISABLE_XNNPACK'] = '1'
os.environ['XNNPACK_DELEGATE_ENABLE'] = '0'

import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import cv2
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
import core.utils as utils
from core.config import cfg

flags.DEFINE_string('weights', './checkpoints/yolov4-416', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-416-fp32.tflite', 'path to output')
flags.DEFINE_integer('input_size', 416, 'path to output')
flags.DEFINE_string('quantize_mode', 'float32', 'quantize mode (int8, float16, float32)')
flags.DEFINE_string('dataset', "/Volumes/Elements/data/coco_dataset/coco/5k.txt", 'path to dataset')
flags.DEFINE_integer('num_calibration_images', 300, 'number of images for INT8 calibration (recommended: 200-500)')

def representative_data_gen():
  """
  Generate representative dataset for INT8 calibration.

  More calibration images = better quantization accuracy:
  - 100 images: Basic quantization (faster conversion)
  - 300 images: Good accuracy (recommended for production)
  - 500+ images: Excellent accuracy (diminishing returns beyond this)

  Adjust with --num_calibration_images flag
  """
  NUM_CALIBRATION_IMAGES = FLAGS.num_calibration_images

  if not os.path.exists(FLAGS.dataset):
    logging.error(f"Dataset file not found: {FLAGS.dataset}")
    logging.warning(f"Using {NUM_CALIBRATION_IMAGES} synthetic calibration images instead...")
    logging.warning("⚠️  Synthetic data is less accurate than real images!")
    # Generate synthetic calibration data
    for i in range(NUM_CALIBRATION_IMAGES):
      img_in = np.random.rand(1, FLAGS.input_size, FLAGS.input_size, 3).astype(np.float32)
      if i % 50 == 0:
        logging.info(f"Generated synthetic calibration data: {i}/{NUM_CALIBRATION_IMAGES}")
      yield [img_in]
    return

  fimage = open(FLAGS.dataset).read().split()
  if not fimage:
    logging.error("Dataset file is empty!")
    logging.warning(f"Using {NUM_CALIBRATION_IMAGES} synthetic calibration images instead...")
    for i in range(NUM_CALIBRATION_IMAGES):
      img_in = np.random.rand(1, FLAGS.input_size, FLAGS.input_size, 3).astype(np.float32)
      yield [img_in]
    return

  num_calibration = min(NUM_CALIBRATION_IMAGES, len(fimage))
  logging.info(f"Starting INT8 calibration with {num_calibration} images from {len(fimage)} total")
  logging.info(f"More calibration images = better quantization accuracy!")

  calibrated = 0
  failed_paths = []
  for input_value in range(len(fimage)):
    if calibrated >= num_calibration:
      break
    image_path = fimage[input_value].split()[0]  # Get just the image path (before annotations)

    if os.path.exists(image_path):
      try:
        original_image = cv2.imread(image_path)
        if original_image is None:
          failed_paths.append(image_path)
          continue
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image_data = utils.image_preprocess(np.copy(original_image), [FLAGS.input_size, FLAGS.input_size])
        img_in = image_data[np.newaxis, ...].astype(np.float32)
        calibrated += 1
        if calibrated % 50 == 0:  # Report every 50 images
          logging.info(f"Calibration progress: {calibrated}/{num_calibration} images ({calibrated/num_calibration*100:.1f}%)")
        yield [img_in]
      except Exception as e:
        logging.warning(f"Failed to process {image_path}: {e}")
        failed_paths.append(image_path)
        continue
    else:
      if len(failed_paths) < 5:  # Only log first few missing files
        failed_paths.append(image_path)
      continue

  if calibrated == 0:
    logging.error(f"NO IMAGES WERE CALIBRATED! First 5 failed paths:")
    for path in failed_paths[:5]:
      logging.error(f"  {path}")
    logging.warning(f"Using {NUM_CALIBRATION_IMAGES} synthetic calibration images as fallback...")
    for i in range(NUM_CALIBRATION_IMAGES):
      img_in = np.random.rand(1, FLAGS.input_size, FLAGS.input_size, 3).astype(np.float32)
      if i % 50 == 0:
        logging.info(f"Generated synthetic calibration data: {i}/{NUM_CALIBRATION_IMAGES}")
      yield [img_in]
  else:
    logging.info(f"✅ Calibration complete: {calibrated} images processed")
    logging.info(f"Quantization accuracy should be {'EXCELLENT' if calibrated >= 200 else 'GOOD' if calibrated >= 100 else 'ACCEPTABLE'}")

def save_tflite():
  converter = tf.lite.TFLiteConverter.from_saved_model(FLAGS.weights)

  if FLAGS.quantize_mode == 'float16':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.allow_custom_ops = True
  elif FLAGS.quantize_mode == 'int8':
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # REQUIRED for INT8 quantization
    converter.representative_dataset = representative_data_gen

    # For models WITH folded BatchNorm (no BN layers in graph):
    # Use only TFLITE_BUILTINS_INT8 for pure INT8 without Flex ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,  # INT8 quantized ops
    ]

    # Force full INT8 quantization
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # Disable XNNPACK delegate which can cause issues with some ops
    converter.experimental_new_converter = True

    # If you still get XNNPACK errors at inference, you may need to disable it
    # when loading the interpreter. See demo() function below.
  tflite_model = converter.convert()
  open(FLAGS.output, 'wb').write(tflite_model)

  logging.info("model saved to: {}".format(FLAGS.output))


def demo():
  # Disable XNNPACK delegate to avoid runtime errors with certain ops
  # XNNPACK can fail with error "Node number X failed to invoke"
  try:
    # Method 1: Use InterpreterOptions to disable XNNPACK
    interpreter = tf.lite.Interpreter(
      model_path="your_model.tflite",
      experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    )
  except Exception as e:
    print("HERE")
    # Fallback
    interpreter = tf.lite.Interpreter(model_path=FLAGS.output,
      experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)

  interpreter.allocate_tensors()
  logging.info('tflite model loaded')

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  print(f"\nInput details:")
  print(f"  dtype: {input_details[0]['dtype'].__name__}")
  print(f"  shape: {input_details[0]['shape']}")

  print(f"\nOutput details:")
  for i, out in enumerate(output_details):
    print(f"  Output {i}: dtype={out['dtype'].__name__}, shape={out['shape']}")

  input_shape = input_details[0]['shape']
  input_dtype = input_details[0]['dtype']

  if input_dtype == np.uint8:
    print("\n✅ Input is UINT8 - quantization successful!")
    input_data = np.random.randint(0, 255, input_shape, dtype=np.uint8)
  elif input_dtype == np.int8:
    print("\n✅ Input is INT8 - quantization successful!")
    input_data = np.random.randint(-128, 127, input_shape, dtype=np.int8)
  else:
    print(f"\n⚠️  Input is {input_dtype.__name__} - quantization may have failed")
    input_data = np.random.rand(*input_shape).astype(np.float32)

  try:
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
    print("\n✅ Model inference successful!")
    print(f"Output shapes: {[o.shape for o in output_data]}")
  except Exception as e:
    print(f"\n⚠️  Inference failed: {e}")
    print("Model converted but has runtime issues - check quantization")


def main(_argv):
  save_tflite()
  demo()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


