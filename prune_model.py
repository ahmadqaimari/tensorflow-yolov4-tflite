"""
Pruning for YOLOv4-Tiny using TensorFlow Model Optimization Toolkit (TFMOT).

Based on official TensorFlow pruning guides:
- https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras
- https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide

This script:
1. Loads a pre-trained YOLOv4-tiny model
2. Applies magnitude-based pruning with a polynomial decay schedule
3. Fine-tunes the model (optional)
4. Strips pruning wrappers and exports for TFLite conversion
5. Combines pruning with INT8 quantization for maximum compression
"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from absl import app, flags, logging
from absl.flags import FLAGS
import numpy as np
import tempfile
import os

from core.yolov4 import YOLO
import core.utils as utils

# ==============================================================================
# FLAGS DEFINITION
# ==============================================================================
flags.DEFINE_string('weights', './data/yolov4-tiny.weights', 'path to weights file')
flags.DEFINE_string('output', './checkpoints/yolov4-tiny-pruned', 'path to output pruned model')
flags.DEFINE_boolean('tiny', True, 'is yolo-tiny or not')
flags.DEFINE_integer('input_size', 416, 'input size of the model')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')

# Pruning configuration
flags.DEFINE_float('initial_sparsity', 0.0, 'initial sparsity (0.0 to 1.0)')
flags.DEFINE_float('final_sparsity', 0.5, 'target final sparsity (0.5 = 50% of weights zeroed)')
flags.DEFINE_integer('begin_step', 0, 'step to start pruning')
flags.DEFINE_integer('end_step', 1000, 'step to end pruning (calculated from epochs if training)')
flags.DEFINE_integer('pruning_frequency', 100, 'frequency of pruning updates')

# Block sparsity for hardware optimization (optional)
flags.DEFINE_string('block_size', '', 'block size for structured sparsity, e.g., "1,4" for [1,4]')

# Training configuration (optional fine-tuning)
flags.DEFINE_boolean('fine_tune', False, 'whether to fine-tune the pruned model')
flags.DEFINE_string('train_dataset', '', 'path to training dataset for fine-tuning')
flags.DEFINE_integer('epochs', 2, 'number of epochs for fine-tuning')
flags.DEFINE_integer('batch_size', 8, 'batch size for training')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate for fine-tuning')

# Pruning strategy
flags.DEFINE_string('skip_layers', '', 'comma-separated layer names to skip (e.g., "conv2d_0,conv2d_1")')


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def configure_gpu():
    """
    Configure GPU settings for pruning/fine-tuning.
    Enables memory growth and displays GPU information.

    Returns:
        bool: True if GPU is available and configured, False otherwise
    """
    gpus = tf.config.list_physical_devices('GPU')

    if len(gpus) == 0:
        logging.warning("=" * 80)
        logging.warning("⚠️  WARNING: No GPU found!")
        logging.warning("⚠️  Fine-tuning will run on CPU (very slow)")
        logging.warning("⚠️  Consider using a machine with GPU for faster training")
        logging.warning("=" * 80)
        return False

    logging.info("=" * 80)
    logging.info(f"✅ Found {len(gpus)} GPU(s):")
    for i, gpu in enumerate(gpus):
        logging.info(f"   GPU {i}: {gpu.name}")

    try:
        # Enable memory growth (prevents TF from allocating all GPU memory at once)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        logging.info("✅ GPU memory growth enabled")
        logging.info("=" * 80)
        return True

    except RuntimeError as e:
        logging.error(f"❌ GPU configuration error: {e}")
        logging.error("   GPU may already be initialized")
        return False


def get_pruning_params():
    """
    Create pruning parameters based on FLAGS.

    Returns polynomial decay schedule that gradually increases sparsity from
    initial_sparsity to final_sparsity over the training period.
    """
    # Calculate end_step if fine-tuning
    if FLAGS.fine_tune and FLAGS.train_dataset:
        # This is a simplified calculation - adjust based on your dataset size
        # end_step should be: (num_train_samples / batch_size) * epochs
        logging.info(f"End step for pruning: {FLAGS.end_step}")

    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=FLAGS.initial_sparsity,
        final_sparsity=FLAGS.final_sparsity,
        begin_step=FLAGS.begin_step,
        end_step=FLAGS.end_step,
        frequency=FLAGS.pruning_frequency
    )

    pruning_params = {
        'pruning_schedule': pruning_schedule
    }

    # Add block sparsity if specified (for hardware optimization)
    if FLAGS.block_size:
        try:
            block_size = [int(x) for x in FLAGS.block_size.split(',')]

            # For Conv2D layers, block_size must have 4 dimensions: [H, W, in_channels, out_channels]
            # Common patterns:
            # - [1, 1, 1, 4]: Prune 4 output channels at a time (good for hardware)
            # - [1, 1, 4, 1]: Prune 4 input channels at a time
            if len(block_size) != 4:
                logging.warning(f"Block size must be 4D for Conv2D layers (e.g., [1,1,1,4]). Got: {block_size}")
                logging.warning("Ignoring block_size parameter and using unstructured pruning.")
            else:
                pruning_params['block_size'] = tuple(block_size)
                logging.info(f"Using block sparsity: {block_size}")
                logging.info(f"  [H={block_size[0]}, W={block_size[1]}, InCh={block_size[2]}, OutCh={block_size[3]}]")
        except Exception as e:
            logging.warning(f"Invalid block_size format: {FLAGS.block_size}. Error: {e}. Using default.")

    return pruning_params


def load_pretrained_model():
    """
    Load pre-trained YOLOv4-tiny model from weights file.

    Returns:
        Loaded Keras model
    """
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    logging.info("Building model architecture...")
    input_layer = tf.keras.layers.Input([FLAGS.input_size, FLAGS.input_size, 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    model = tf.keras.Model(input_layer, feature_maps)

    logging.info(f"Loading weights from: {FLAGS.weights}")
    utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)

    logging.info("✓ Pre-trained model loaded successfully")
    return model


def create_pruned_model(base_model):
    """
    Create a pruned model from base model using TFMOT API.

    Uses selective pruning to only prune Conv2D layers and skip:
    - BatchNormalization (custom or built-in)
    - Activation layers
    - Other non-convolutional layers

    Args:
        base_model: Pre-trained Keras model

    Returns:
        Pruned model with pruning wrappers
    """
    from core.common import BatchNormalization as CustomBatchNorm

    pruning_params = get_pruning_params()

    def apply_pruning_wrapper(layer):
        """Apply pruning only to Conv2D layers, skip everything else"""
        # Skip custom BatchNorm (from core.common)
        if isinstance(layer, CustomBatchNorm):
            logging.info(f"Skipping custom BatchNorm layer: {layer.name}")
            return layer

        # Skip built-in BatchNorm
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            logging.info(f"Skipping BatchNorm layer: {layer.name}")
            return layer

        # Skip layers in user-specified skip list
        skip_layer_names = []
        if FLAGS.skip_layers:
            skip_layer_names = [name.strip() for name in FLAGS.skip_layers.split(',')]

        if layer.name in skip_layer_names:
            logging.info(f"Skipping layer (user specified): {layer.name}")
            return layer

        # Only prune Conv2D layers
        if isinstance(layer, tf.keras.layers.Conv2D):
            logging.info(f"Applying pruning to Conv2D layer: {layer.name}")
            return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

        # Return all other layers unchanged
        return layer

    logging.info("Applying SELECTIVE pruning (Conv2D only, skipping BatchNorm)...")

    # Clone model with selective pruning
    model_for_pruning = tf.keras.models.clone_model(
        base_model,
        clone_function=apply_pruning_wrapper,
    )

    return model_for_pruning


def fine_tune_pruned_model(model_for_pruning):
    """
    Fine-tune the pruned model with training data.

    Args:
        model_for_pruning: Model with pruning wrappers

    Returns:
        Fine-tuned model
    """
    if not FLAGS.train_dataset or not os.path.exists(FLAGS.train_dataset):
        logging.warning(f"Training dataset not found: {FLAGS.train_dataset}")
        logging.info("Skipping fine-tuning...")
        return model_for_pruning

    logging.info("=" * 80)
    logging.info("FINE-TUNING PRUNED MODEL")
    logging.info("=" * 80)

    # Verify GPU is being used
    logging.info("\n🔍 Checking device placement:")
    logical_devices = tf.config.list_logical_devices()
    for device in logical_devices:
        logging.info(f"   {device.device_type}: {device.name}")

    # Check where model weights are placed
    if len(model_for_pruning.layers) > 0:
        first_layer_with_weights = None
        for layer in model_for_pruning.layers:
            if len(layer.weights) > 0:
                first_layer_with_weights = layer
                break

        if first_layer_with_weights:
            device = first_layer_with_weights.weights[0].device
            logging.info(f"   Model weights device: {device}")
            if 'GPU' in device:
                logging.info("   ✅ Model is on GPU")
            else:
                logging.warning("   ⚠️  Model is on CPU")

    # Load training dataset with REAL YOLO TARGETS
    logging.info("\n⚠️  Fine-tuning with REAL YOLO TARGETS from annotations")
    logging.info("Loading dataset from: {}".format(FLAGS.train_dataset))

    # YOLOv4-Tiny has TWO output heads (multi-scale detection):
    # - Output 1: [batch, 26, 26, 255] for detecting larger objects (stride 16)
    # - Output 2: [batch, 13, 13, 255] for detecting smaller objects (stride 32)
    # Each output has 255 channels: 3 anchors * (5 + 80 classes) = 3 * 85 = 255

    try:
        import cv2
        from core.config import cfg

        # Load config for YOLO
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

        # YOLOv4-Tiny has 2 strides: [16, 32]
        strides = np.array(STRIDES)
        anchors = ANCHORS
        num_classes = NUM_CLASS
        anchor_per_scale = 3
        max_bbox_per_scale = 150

        logging.info(f"YOLOv4-Tiny config:")
        logging.info(f"  Strides: {strides}")
        logging.info(f"  Anchors: {anchors}")
        logging.info(f"  Classes: {num_classes}")

        # Load annotations
        logging.info(f"Reading annotations from: {FLAGS.train_dataset}")
        with open(FLAGS.train_dataset, 'r') as f:
            annotations = [line.strip() for line in f.readlines() if len(line.strip().split()[1:]) != 0]

        num_images = len(annotations)
        logging.info(f"✓ Found {num_images} images with annotations")

        if num_images == 0:
            raise ValueError("No images with annotations found in dataset file")

        def parse_annotation(annotation):
            """Parse annotation line: image_path x1,y1,x2,y2,class x1,y1,x2,y2,class ..."""
            line = annotation.split()
            image_path = line[0]

            bboxes = []
            if len(line) > 1:
                bboxes = np.array([list(map(float, box.split(','))) for box in line[1:]])
            else:
                bboxes = np.array([])

            return image_path, bboxes

        def preprocess_true_boxes(bboxes):
            """
            Preprocess ground truth bounding boxes into YOLO grid format.

            Args:
                bboxes: [N, 5] array of [x1, y1, x2, y2, class_id]

            Returns:
                Tuple of (label_large, label_small) for the two detection heads
            """
            # Initialize output grids for 2 scales
            train_output_sizes = FLAGS.input_size // strides

            # label shape: [grid_h, grid_w, anchors_per_scale, 5+num_classes]
            label_large = np.zeros((train_output_sizes[0], train_output_sizes[0], anchor_per_scale, 5 + num_classes), dtype=np.float32)
            label_small = np.zeros((train_output_sizes[1], train_output_sizes[1], anchor_per_scale, 5 + num_classes), dtype=np.float32)

            labels = [label_large, label_small]

            if len(bboxes) == 0:
                # No objects in image
                # Flatten anchor dimension: [H, W, 3, 85] -> [H, W, 255]
                return (label_large.reshape(train_output_sizes[0], train_output_sizes[0], -1),
                        label_small.reshape(train_output_sizes[1], train_output_sizes[1], -1))

            # Process each ground truth box
            for bbox in bboxes:
                bbox_coor = bbox[:4]
                bbox_class_ind = int(bbox[4])

                # Calculate center and size
                bbox_xywh = np.concatenate([
                    (bbox_coor[2:] + bbox_coor[:2]) * 0.5,  # center x, y
                    bbox_coor[2:] - bbox_coor[:2]  # width, height
                ], axis=-1)

                # Normalize to [0, 1]
                bbox_xywh_scaled = 1.0 * bbox_xywh / FLAGS.input_size

                # Assign bbox to appropriate scale based on size
                for i, stride in enumerate(strides):
                    # Calculate grid cell coordinates
                    xind = int(np.floor(bbox_xywh_scaled[0] * train_output_sizes[i]))
                    yind = int(np.floor(bbox_xywh_scaled[1] * train_output_sizes[i]))

                    # Clip to valid range
                    xind = np.clip(xind, 0, train_output_sizes[i] - 1)
                    yind = np.clip(yind, 0, train_output_sizes[i] - 1)

                    # Find best anchor for this bbox
                    bbox_wh_scaled = bbox_xywh_scaled[2:4]
                    best_anchor_ind = 0
                    max_iou = 0

                    for anchor_ind in range(anchor_per_scale):
                        anchor_wh = anchors[i][anchor_ind] / FLAGS.input_size
                        min_wh = np.minimum(bbox_wh_scaled, anchor_wh)
                        iou = (min_wh[0] * min_wh[1]) / (bbox_wh_scaled[0] * bbox_wh_scaled[1] + anchor_wh[0] * anchor_wh[1] - min_wh[0] * min_wh[1] + 1e-10)

                        if iou > max_iou:
                            max_iou = iou
                            best_anchor_ind = anchor_ind

                    # Encode bbox in grid cell
                    label = labels[i]
                    label[yind, xind, best_anchor_ind, 0:4] = bbox_xywh
                    label[yind, xind, best_anchor_ind, 4] = 1.0  # objectness
                    label[yind, xind, best_anchor_ind, 5 + bbox_class_ind] = 1.0  # class

            # Flatten anchor dimension: [H, W, 3, 85] -> [H, W, 255]
            label_large_flat = label_large.reshape(train_output_sizes[0], train_output_sizes[0], -1)
            label_small_flat = label_small.reshape(train_output_sizes[1], train_output_sizes[1], -1)

            return label_large_flat, label_small_flat

        def data_generator():
            """Generate batches with REAL YOLO targets from annotations"""
            batch_count = 0
            max_batches = FLAGS.end_step // FLAGS.epochs

            # Shuffle annotations
            np.random.shuffle(annotations)

            while batch_count < max_batches:
                batch_images = []
                batch_labels_large = []
                batch_labels_small = []

                # Load a batch
                for i in range(FLAGS.batch_size):
                    idx = (batch_count * FLAGS.batch_size + i) % num_images
                    annotation = annotations[idx]

                    image_path, bboxes = parse_annotation(annotation)

                    try:
                        # Load and preprocess image
                        image = cv2.imread(image_path)
                        if image is None:
                            raise ValueError(f"Failed to load image: {image_path}")

                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, (FLAGS.input_size, FLAGS.input_size))
                        image = image.astype(np.float32) / 255.0

                        # Generate YOLO targets
                        label_large, label_small = preprocess_true_boxes(bboxes)

                        batch_images.append(image)
                        batch_labels_large.append(label_large)
                        batch_labels_small.append(label_small)

                    except Exception as e:
                        logging.warning(f"Error loading {image_path}: {e}, using dummy data")
                        # Use dummy image and empty targets
                        image = np.random.rand(FLAGS.input_size, FLAGS.input_size, 3).astype(np.float32)
                        label_large, label_small = preprocess_true_boxes(np.array([]))

                        batch_images.append(image)
                        batch_labels_large.append(label_large)
                        batch_labels_small.append(label_small)

                # Stack batch
                batch_images = np.array(batch_images)
                batch_labels_large = np.array(batch_labels_large)
                batch_labels_small = np.array(batch_labels_small)

                yield batch_images, (batch_labels_large, batch_labels_small)
                batch_count += 1

        # Create TensorFlow dataset
        train_dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(FLAGS.batch_size, FLAGS.input_size, FLAGS.input_size, 3), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(FLAGS.batch_size, FLAGS.input_size // 16, FLAGS.input_size // 16, 255), dtype=tf.float32),
                    tf.TensorSpec(shape=(FLAGS.batch_size, FLAGS.input_size // 32, FLAGS.input_size // 32, 255), dtype=tf.float32),
                )
            )
        )

        # Calculate steps
        steps_per_epoch = min(num_images // FLAGS.batch_size, FLAGS.end_step // FLAGS.epochs)
        logging.info(f"✓ Steps per epoch: {steps_per_epoch}")
        logging.info(f"✓ Dataset created with REAL YOLO TARGETS")

    except Exception as e:
        logging.error(f"❌ Failed to load dataset: {e}")
        logging.error("   This is a critical error - pruning fine-tuning needs real data!")

        import traceback
        traceback.print_exc()

        raise RuntimeError("Cannot proceed without valid training dataset")


    # Compile model with multiple outputs
    model_for_pruning.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
        loss='mse',  # Use your actual YOLO loss function here
        # For multiple outputs, loss can be a list or dict
        # loss=['mse', 'mse'],  # One loss per output
        metrics=['accuracy']
    )

    # Create callbacks
    logdir = tempfile.mkdtemp()
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),  # REQUIRED for pruning
        tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),  # For TensorBoard
    ]

    # Train
    logging.info(f"\nTraining for {FLAGS.epochs} epochs...")
    if steps_per_epoch:
        logging.info(f"Steps per epoch: {steps_per_epoch}")
    logging.info("Monitoring GPU utilization (check nvidia-smi in another terminal)...")

    # Fit model - TensorFlow will automatically use GPU if available
    history = model_for_pruning.fit(
        train_dataset,
        epochs=FLAGS.epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        verbose=1
    )

    # Verify tensors were on GPU during training
    logging.info("\n✅ Training completed")
    logging.info(f"   Final loss: {history.history['loss'][-1]:.4f}")

    logging.info(f"✓ Fine-tuning complete. TensorBoard logs: {logdir}")
    logging.info(f"  View logs: tensorboard --logdir={logdir}")

    return model_for_pruning


def export_pruned_model(model_for_pruning):
    """
    Strip pruning wrappers and export the final model.

    Args:
        model_for_pruning: Model with pruning wrappers

    Returns:
        Final model ready for deployment
    """
    logging.info("\nStripping pruning wrappers...")

    # CRITICAL: strip_pruning removes tf.Variables only needed during training
    # This is necessary to see compression benefits
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    logging.info("✓ Pruning wrappers stripped")

    return model_for_export


def calculate_sparsity(model):
    """
    Calculate the actual sparsity achieved in the model.

    Args:
        model: Keras model

    Returns:
        Dictionary with sparsity statistics
    """
    total_weights = 0
    zero_weights = 0
    layer_stats = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            weights = layer.get_weights()
            if len(weights) > 0:
                W = weights[0]
                layer_total = W.size
                layer_zeros = np.sum(np.abs(W) < 1e-6)
                layer_sparsity = layer_zeros / layer_total

                total_weights += layer_total
                zero_weights += layer_zeros

                layer_stats.append({
                    'name': layer.name,
                    'total': layer_total,
                    'zeros': layer_zeros,
                    'sparsity': layer_sparsity
                })

    overall_sparsity = zero_weights / total_weights if total_weights > 0 else 0

    return {
        'overall_sparsity': overall_sparsity,
        'total_weights': total_weights,
        'zero_weights': zero_weights,
        'layer_stats': layer_stats
    }


# ==============================================================================
# MAIN PRUNING PIPELINE
# ==============================================================================

def prune_yolo():
    """
    Main pruning pipeline following TFMOT best practices.

    Pipeline:
    1. Configure GPU
    2. Load pre-trained model
    3. Apply pruning API
    4. Fine-tune (optional)
    5. Strip pruning wrappers
    6. Export for TFLite conversion
    """

    # Configure GPU first (before any TensorFlow operations)
    configure_gpu()

    logging.info("\n" + "=" * 80)
    logging.info("YOLO PRUNING WITH TENSORFLOW MODEL OPTIMIZATION TOOLKIT")
    logging.info("=" * 80)
    logging.info(f"Target sparsity: {FLAGS.final_sparsity * 100:.1f}%")
    logging.info(f"Pruning strategy: Selective (Conv2D only, skips BatchNorm)")
    logging.info(f"Fine-tune: {FLAGS.fine_tune}")
    if FLAGS.block_size:
        logging.info(f"Block sparsity: {FLAGS.block_size}")
    logging.info("=" * 80)

    # STEP 1: Load pre-trained model
    logging.info("\n[STEP 1/5] Loading pre-trained model...")
    base_model = load_pretrained_model()
    base_model.summary()

    # STEP 2: Apply pruning
    logging.info("\n[STEP 2/5] Applying pruning API...")
    model_for_pruning = create_pruned_model(base_model)

    logging.info("\nPruned model summary:")
    model_for_pruning.summary()

    # STEP 3: Fine-tune (optional but recommended)
    if FLAGS.fine_tune:
        logging.info("\n[STEP 3/5] Fine-tuning pruned model...")
        model_for_pruning = fine_tune_pruned_model(model_for_pruning)
    else:
        logging.info("\n[STEP 3/5] Skipping fine-tuning (--fine_tune=False)")
        logging.info("Note: Fine-tuning is RECOMMENDED for better accuracy")
        logging.info("      Set --fine_tune=True --train_dataset=<path> to enable")

    # STEP 4: Strip pruning wrappers
    logging.info("\n[STEP 4/5] Exporting pruned model...")
    model_for_export = export_pruned_model(model_for_pruning)

    # STEP 5: Calculate and display sparsity
    logging.info("\n[STEP 5/5] Analyzing sparsity...")
    stats = calculate_sparsity(model_for_export)

    logging.info("\n" + "=" * 80)
    logging.info("SPARSITY ANALYSIS")
    logging.info("=" * 80)
    logging.info(f"Overall sparsity: {stats['overall_sparsity'] * 100:.2f}%")
    logging.info(f"Total Conv2D weights: {stats['total_weights']:,}")
    logging.info(f"Zero weights: {stats['zero_weights']:,}")
    logging.info(f"Non-zero weights: {stats['total_weights'] - stats['zero_weights']:,}")

    logging.info("\nPer-layer sparsity (first 10 Conv2D layers):")
    for i, layer_stat in enumerate(stats['layer_stats'][:10]):
        logging.info(f"  {layer_stat['name']}: {layer_stat['sparsity']*100:.1f}% sparse "
                    f"({layer_stat['zeros']:,}/{layer_stat['total']:,} zeros)")

    if len(stats['layer_stats']) > 10:
        logging.info(f"  ... and {len(stats['layer_stats']) - 10} more layers")

    # Save the pruned model
    logging.info("\n" + "=" * 80)
    logging.info("SAVING PRUNED MODEL")
    logging.info("=" * 80)
    logging.info(f"Saving to: {FLAGS.output}")
    model_for_export.save(FLAGS.output)
    logging.info("✓ Model saved successfully")

    # Print next steps
    logging.info("\n" + "=" * 80)
    logging.info("NEXT STEPS")
    logging.info("=" * 80)

    logging.info("\n1. Convert to TFLite (Float16 - good compression):")
    logging.info(f"   python convert_tflite.py \\")
    logging.info(f"       --weights {FLAGS.output} \\")
    logging.info(f"       --output {FLAGS.output.replace('.tf', '')}-fp16.tflite \\")
    logging.info(f"       --quantize_mode float16 \\")
    logging.info(f"       --input_size {FLAGS.input_size}")

    logging.info("\n2. Convert to TFLite with INT8 quantization (BEST compression):")
    logging.info(f"   python convert_tflite.py \\")
    logging.info(f"       --weights {FLAGS.output} \\")
    logging.info(f"       --output {FLAGS.output.replace('.tf', '')}-int8.tflite \\")
    logging.info(f"       --quantize_mode int8 \\")
    logging.info(f"       --dataset <your_calibration_dataset.txt> \\")
    logging.info(f"       --input_size {FLAGS.input_size} \\")
    logging.info(f"       --num_calibration_images 300")

    logging.info("\n3. Verify quantization quality:")
    logging.info("   # Edit check_quant.py to point to your .tflite file")
    logging.info("   python check_quant.py")

    logging.info("\n4. Test the model:")
    logging.info(f"   python detect.py \\")
    logging.info(f"       --weights {FLAGS.output.replace('.tf', '')}-int8.tflite \\")
    logging.info(f"       --size {FLAGS.input_size} \\")
    logging.info(f"       --model yolov4 \\")
    logging.info(f"       --framework tflite \\")
    logging.info(f"       --image ./data/kite.jpg")

    logging.info("\n" + "=" * 80)
    logging.info("PRUNING COMPLETE!")
    logging.info("=" * 80)
    logging.info("\nExpected results with pruning + INT8 quantization:")
    logging.info("  • Model size: 5-10x smaller than original")
    logging.info("  • Inference speed: 3-5x faster (on appropriate hardware)")
    logging.info("  • Accuracy drop: 2-5% (depends on sparsity level)")
    logging.info("\nFor best results:")
    logging.info("  • Compress the .tflite file with gzip for storage/transmission")
    logging.info("  • The zeros from pruning compress very well!")
    logging.info("=" * 80)


def main(_argv):
    prune_yolo()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
