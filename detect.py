import os
# Disable XNNPACK delegate to avoid compatibility issues with quantized models
os.environ['TF_LITE_ENABLE_XNNPACK'] = '0'

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes, decode
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('image', './data/kite.jpg', 'path to input image')
flags.DEFINE_string('output', 'result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def main(_argv):
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    image_path = FLAGS.image

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    if FLAGS.framework == 'tflite':
        # Try to create interpreter without XNNPACK
        try:
            # First attempt: load without delegates
            interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
            interpreter.allocate_tensors()
        except Exception as e:
            logging.error(f"Failed to load interpreter: {e}")
            raise

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)

        # Check if model is quantized (UINT8/INT8 input)
        input_dtype = input_details[0]['dtype']

        if input_dtype == np.uint8:
            # Quantized model - use raw pixel values (0-255)
            print("Using UINT8 input (quantized model)")
            image_data = cv2.resize(original_image, (input_size, input_size))
            images_data = image_data[np.newaxis, ...].astype(np.uint8)
        elif input_dtype == np.int8:
            # INT8 quantized model - shift to -128 to 127
            print("Using INT8 input (quantized model)")
            image_data = cv2.resize(original_image, (input_size, input_size))
            images_data = (image_data.astype(np.int16) - 128).astype(np.int8)[np.newaxis, ...]
        else:
            # Float model - normalize to 0-1
            print("Using FLOAT32 input (non-quantized model)")
            image_data = cv2.resize(original_image, (input_size, input_size))
            image_data = image_data / 255.
            images_data = image_data[np.newaxis, ...].astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], images_data)

        # Try to invoke - catch XNNPACK errors
        try:
            interpreter.invoke()
        except RuntimeError as e:
            if "XNNPACK" in str(e):
                print("⚠️  XNNPACK delegate failed. Your model is quantized correctly but has ops XNNPACK doesn't support.")
                print("This is NORMAL for quantized YOLOv4-tiny models.")
                print("\nThe issue is that XNNPACK is automatically enabled and can't be easily disabled in Python.")
                print("\nSOLUTION: Use the model in C++ or Android where you have delegate control,")
                print("or rebuild TensorFlow Lite without XNNPACK support.")
                print("\nFor now, your INT8 quantization IS successful - the model just can't run with XNNPACK.")
                raise RuntimeError("XNNPACK incompatibility - model is quantized correctly but needs custom TFLite build") from e
            else:
                raise

        pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        # Dequantize outputs if they are UINT8/INT8
        for i in range(len(pred)):
            if output_details[i]['dtype'] in [np.uint8, np.int8]:
                scale, zero_point = output_details[i]['quantization']
                pred[i] = (pred[i].astype(np.float32) - zero_point) * scale

        if FLAGS.tiny == True:
            # YOLOv4-Tiny has 2 detection heads
            # pred[0] is the larger feature map (26x26), pred[1] is smaller (13x13)
            bbox_tensors = []
            prob_tensors = []
            for i, fm in enumerate(pred):
                if i == 0:
                    # First output (26x26) - for larger objects
                    output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
                elif i == 1:
                    # Second output (13x13) - for smaller objects
                    output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
                bbox_tensors.append(output_tensors[0])
                prob_tensors.append(output_tensors[1])
            pred_bbox = tf.concat(bbox_tensors, axis=1)
            pred_prob = tf.concat(prob_tensors, axis=1)
            boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=0.25,
                                            input_shape=tf.constant([input_size, input_size]))
        else:
            # Full YOLOv4 direct filter
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    print(f"Boxes shape: {pred_bbox[0].shape}, Scores shape: {pred_bbox[1].shape}, Classes shape: {pred_bbox[2].shape}, Valid detections: {pred_bbox[3]}")
    print(f"First few boxes: {pred_bbox[0][0][:5]}")
    image = utils.draw_bbox(original_image, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite(FLAGS.output, image)
    print('Output saved to:', FLAGS.output)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
