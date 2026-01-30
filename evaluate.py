from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
from core.yolov4 import filter_boxes, decode
from tensorflow.python.saved_model import tag_constants
import core.utils as utils
from core.config import cfg

flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_string('framework', 'tf', 'select model type in (tf, tflite, trt)'
                    'path to weights file')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('annotation_path', "./data/dataset/val2017.txt", 'annotation path')
flags.DEFINE_string('write_image_path', "./data/detection/", 'write image path')
flags.DEFINE_float('iou', 0.5, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def main(_argv):
    INPUT_SIZE = FLAGS.size
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)

    predicted_dir_path = './mAP/predicted'
    ground_truth_dir_path = './mAP/ground-truth'
    if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
    if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
    if os.path.exists(cfg.TEST.DECTECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DECTECTED_IMAGE_PATH)

    os.mkdir(predicted_dir_path)
    os.mkdir(ground_truth_dir_path)
    os.mkdir(cfg.TEST.DECTECTED_IMAGE_PATH)

    # Build Model
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    num_lines = sum(1 for line in open(FLAGS.annotation_path))
    print(f"Starting evaluation on {num_lines} images...")

    processed = 0
    skipped = 0

    with open(FLAGS.annotation_path, 'r') as annotation_file:  # Use FLAGS.annotation_path, not cfg.TEST.ANNOT_PATH
        for num, line in enumerate(annotation_file):
            annotation = line.strip().split()
            image_path = annotation[0]

            # Normalize path separators for Windows
            image_path = image_path.replace('/', os.sep).replace('\\', os.sep)

            # Check if image exists
            if not os.path.exists(image_path):
                print(f'=> Skipping {image_path} (file not found)')
                skipped += 1
                continue

            image_name = os.path.basename(image_path)
            image = cv2.imread(image_path)

            # Check if image was loaded successfully
            if image is None:
                print(f'=> Skipping {image_path} (failed to load)')
                skipped += 1
                continue

            processed += 1

            # Show progress every 100 images
            if processed % 100 == 0:
                print(f"Progress: {processed}/{num_lines} images processed ({processed/num_lines*100:.1f}%), {skipped} skipped")

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

            if len(bbox_data_gt) == 0:
                bboxes_gt = []
                classes_gt = []
            else:
                bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
            ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

            print('=> ground truth of %s:' % image_name)
            num_bbox_gt = len(bboxes_gt)
            with open(ground_truth_path, 'w') as f:
                for i in range(num_bbox_gt):
                    class_name = CLASSES[classes_gt[i]]
                    xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                    bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print('=> predict result of %s:' % image_name)
            predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
            # Predict Process
            image_size = image.shape[:2]

            if FLAGS.framework == 'tflite':
                # Check if model is quantized (UINT8/INT8 input)
                input_dtype = input_details[0]['dtype']

                if input_dtype == np.uint8:
                    # Quantized model - use raw pixel values (0-255)
                    image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
                    image_data = image_data[np.newaxis, ...].astype(np.uint8)
                elif input_dtype == np.int8:
                    # INT8 quantized model - shift to -128 to 127
                    image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
                    image_data = (image_data.astype(np.int16) - 128).astype(np.int8)[np.newaxis, ...]
                else:
                    # Float model - normalize to 0-1
                    image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
                    image_data = image_data / 255.
                    image_data = image_data[np.newaxis, ...].astype(np.float32)

                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

                # Dequantize outputs if they are UINT8/INT8
                for i in range(len(pred)):
                    if output_details[i]['dtype'] in [np.uint8, np.int8]:
                        scale, zero_point = output_details[i]['quantization']
                        pred[i] = (pred[i].astype(np.float32) - zero_point) * scale

                # Sort predictions by spatial size (larger grid first for tiny model)
                # YOLOv4-tiny outputs: [26x26, 13x13] or [13x13, 26x26] depending on model
                if len(pred) == 2:
                    # Check the spatial dimensions and sort
                    if pred[0].shape[1] < pred[1].shape[1]:  # If first output is smaller
                        pred = [pred[1], pred[0]]  # Swap to [larger, smaller]

                # After sorting: pred[0] is always 26x26, pred[1] is always 13x13
                # Use decode() like detect.py does for proper anchor-based decoding
                if FLAGS.model == 'yolov4' and FLAGS.tiny == True:
                    # YOLOv4-Tiny: decode both output heads with proper anchors
                    bbox_tensors = []
                    prob_tensors = []

                    for i, fm in enumerate(pred):
                        if i == 0:
                            # First output (26x26) - stride 16
                            output_tensors = decode(fm, INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
                        elif i == 1:
                            # Second output (13x13) - stride 32
                            output_tensors = decode(fm, INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, 'tflite')
                        bbox_tensors.append(output_tensors[0])
                        prob_tensors.append(output_tensors[1])

                    # Concatenate both heads
                    pred_bbox = tf.concat(bbox_tensors, axis=1)
                    pred_prob = tf.concat(prob_tensors, axis=1)

                    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=0.25,
                                                   input_shape=tf.constant([INPUT_SIZE, INPUT_SIZE]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25)
            else:
                # TensorFlow model - use float32 normalized input
                image_data = cv2.resize(np.copy(image), (INPUT_SIZE, INPUT_SIZE))
                image_data = image_data / 255.
                image_data = image_data[np.newaxis, ...].astype(np.float32)

                batch_data = tf.constant(image_data)
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
            boxes, scores, classes, valid_detections = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]

            # if cfg.TEST.DECTECTED_IMAGE_PATH is not None:
            #     image_result = utils.draw_bbox(np.copy(image), [boxes, scores, classes, valid_detections])
            #     cv2.imwrite(cfg.TEST.DECTECTED_IMAGE_PATH + image_name, image_result)

            with open(predict_result_path, 'w') as f:
                image_h, image_w, _ = image.shape
                for i in range(valid_detections[0]):
                    if int(classes[0][i]) < 0 or int(classes[0][i]) > NUM_CLASS: continue
                    coor = boxes[0][i]
                    coor[0] = int(coor[0] * image_h)
                    coor[2] = int(coor[2] * image_h)
                    coor[1] = int(coor[1] * image_w)
                    coor[3] = int(coor[3] * image_w)

                    score = scores[0][i]
                    class_ind = int(classes[0][i])
                    class_name = CLASSES[class_ind]
                    score = '%.4f' % score
                    ymin, xmin, ymax, xmax = list(map(str, coor))
                    bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                    f.write(bbox_mess)
                    print('\t' + str(bbox_mess).strip())
            print(num, num_lines)

    # Print final summary
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print(f"Total images processed: {processed}/{num_lines}")
    print(f"Skipped images: {skipped}")
    print(f"Predictions saved to: {predicted_dir_path}")
    print(f"Ground truth saved to: {ground_truth_dir_path}")
    print("\nNext steps:")
    print("  cd mAP/extra")
    print("  python remove_space.py")
    print("  cd ..")
    print("  python main.py --output results_int8")
    print("="*80)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass


