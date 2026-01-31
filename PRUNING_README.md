# YOLOv4-Tiny Pruning with TFMOT

## Overview

This implementation uses **TensorFlow Model Optimization Toolkit (TFMOT)** to apply magnitude-based weight pruning to YOLOv4-Tiny models. The implementation follows official TensorFlow guidelines and best practices.

## Quick Start

```bash
# 1. Prune the model (50% sparsity)
python prune_model.py \
    --weights ./data/yolov4-tiny.weights \
    --output ./checkpoints/yolov4-tiny-pruned \
    --model yolov4 --tiny \
    --final_sparsity 0.5

# 2. Convert to TFLite with INT8 quantization
python convert_tflite.py \
    --weights ./checkpoints/yolov4-tiny-pruned \
    --output ./checkpoints/yolov4-tiny-pruned-int8.tflite \
    --quantize_mode int8 \
    --dataset ./data/dataset/val2017.txt \
    --input_size 416

# 3. Test the pruned model
python detect.py \
    --weights ./checkpoints/yolov4-tiny-pruned-int8.tflite \
    --size 416 --framework tflite \
    --image ./data/kite.jpg
```

## What is Pruning?

**Magnitude-based weight pruning** gradually zeros out model weights based on their magnitude. The key benefits are:

- **Smaller models**: 3-10x compression (with gzip)
- **Faster inference**: On hardware supporting sparse operations
- **Minimal accuracy loss**: With proper configuration
- **Combines with quantization**: For maximum compression

## Pruning Pipeline

The pruning process follows these steps:

```
1. Load Pre-trained Model
   ↓
2. Apply Pruning API (wrap layers with pruning logic)
   ↓
3. [Optional] Fine-tune with UpdatePruningStep callback
   ↓
4. Strip Pruning Wrappers (remove training-only variables)
   ↓
5. Export SavedModel
   ↓
6. Convert to TFLite with INT8 Quantization
   ↓
7. [Optional] Compress with gzip
```

## Configuration Parameters

### Core Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--weights` | Path to YOLOv4-tiny weights | Required | - |
| `--output` | Output path for pruned model | Required | - |
| `--final_sparsity` | Target sparsity (0.0-1.0) | 0.5 | 0.3-0.7 |
| `--input_size` | Input image size | 416 | 416 |

### Pruning Schedule

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--initial_sparsity` | Starting sparsity | 0.0 |
| `--begin_step` | Step to start pruning | 0 |
| `--end_step` | Step to finish pruning | 1000 |
| `--pruning_frequency` | Pruning update frequency | 100 |

### Advanced Options

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--block_size` | Structured sparsity | `"1,16"` |
| `--prune_whole_model` | Prune all vs selective | True |
| `--skip_layers` | Layers to skip | `"conv2d_0,conv2d_20"` |

### Fine-Tuning (Optional)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--fine_tune` | Enable fine-tuning | False |
| `--train_dataset` | Training data path | - |
| `--epochs` | Training epochs | 2 |
| `--batch_size` | Batch size | 8 |
| `--learning_rate` | Learning rate | 1e-4 |

## Usage Examples

### Example 1: Basic Pruning (50% Sparsity)

```bash
python prune_model.py \
    --weights ./data/yolov4-tiny.weights \
    --output ./checkpoints/yolov4-tiny-pruned \
    --model yolov4 --tiny \
    --final_sparsity 0.5
```

**Expected**: ~2-3% accuracy drop, 3-5x compression with quantization

### Example 2: Aggressive Pruning (80% Sparsity)

```bash
python prune_model.py \
    --weights ./data/yolov4-tiny.weights \
    --output ./checkpoints/yolov4-tiny-pruned-80 \
    --model yolov4 --tiny \
    --final_sparsity 0.8
```

**Expected**: ~5-8% accuracy drop, 5-10x compression with quantization

### Example 3: Selective Layer Pruning

```bash
python prune_model.py \
    --weights ./data/yolov4-tiny.weights \
    --output ./checkpoints/yolov4-tiny-pruned-selective \
    --model yolov4 --tiny \
    --final_sparsity 0.5 \
    --prune_whole_model False \
    --skip_layers "conv2d_0,conv2d_1"
```

**Better accuracy** with slightly less compression by protecting critical layers.

### Example 4: Block Sparsity for Hardware

```bash
python prune_model.py \
    --weights ./data/yolov4-tiny.weights \
    --output ./checkpoints/yolov4-tiny-pruned-block \
    --model yolov4 --tiny \
    --final_sparsity 0.5 \
    --block_size "1,16"
```

Optimized for CPUs with 128-bit registers + INT8 (e.g., ARM with NEON).

### Example 5: With Fine-Tuning

```bash
# Best practice: Use separate training dataset
python prune_model.py \
    --weights ./data/yolov4-tiny.weights \
    --output ./checkpoints/yolov4-tiny-pruned-ft \
    --model yolov4 --tiny \
    --final_sparsity 0.5 \
    --fine_tune True \
    --train_dataset ./data/dataset/train2017.txt \
    --epochs 2 \
    --batch_size 8

# Alternative: Use validation set if training data unavailable (not ideal)
# Note: Using val2017.txt for fine-tuning may cause overfitting to your test set
python prune_model.py \
    --weights ./data/yolov4-tiny.weights \
    --output ./checkpoints/yolov4-tiny-pruned-ft \
    --model yolov4 --tiny \
    --final_sparsity 0.5 \
    --fine_tune True \
    --train_dataset ./data/dataset/val2017.txt \
    --epochs 1 \
    --batch_size 8
```

**Minimal accuracy drop** with proper fine-tuning. 

**⚠️ Important Note on Fine-Tuning Data:**
- **Best practice**: Use a separate training dataset (e.g., `train2017.txt`)
- **If no training data**: You can use `val2017.txt`, but:
  - Use fewer epochs (1 instead of 2) to reduce overfitting
  - Your final accuracy metrics may be overly optimistic
  - Consider splitting your validation set: 80% for fine-tuning, 20% for testing

### Example 6: Block Sparsity + Fine-Tuning (Best of Both Worlds)

```bash
# For DE1-SoC with FPGA systolic arrays (Conv2D on FPGA, other layers on ARM CPU)
# Use block_size=[4,4] to match FPGA systolic array architecture

# Option A: With separate training dataset (recommended)
python prune_model.py \
    --weights ./data/yolov4-tiny.weights \
    --output ./checkpoints/yolov4-tiny-pruned-fpga-ft \
    --model yolov4 --tiny \
    --final_sparsity 0.5 \
    --block_size "4,4" \
    --fine_tune True \
    --train_dataset ./data/dataset/train2017.txt \
    --epochs 2 \
    --batch_size 8 \
    --learning_rate 1e-4

# Option B: Using val2017.txt if training data unavailable (reduce epochs to avoid overfitting)
python prune_model.py \
    --weights ./data/yolov4-tiny.weights \
    --output ./checkpoints/yolov4-tiny-pruned-fpga-ft \
    --model yolov4 --tiny \
    --final_sparsity 0.5 \
    --block_size "4,4" \
    --fine_tune True \
    --train_dataset ./data/dataset/val2017.txt \
    --epochs 1 \
    --batch_size 8 \
    --learning_rate 1e-4

# Then convert to TFLite for heterogeneous execution:
# - Conv2D layers → FPGA fabric (OpenCL systolic arrays benefit from 4×4 blocks)
# - BatchNorm, activations, other layers → ARM Cortex-A9 CPU
python convert_tflite.py \
    --weights ./checkpoints/yolov4-tiny-pruned-fpga-ft \
    --output ./checkpoints/yolov4-tiny-pruned-fpga-int8.tflite \
    --quantize_mode int8 \
    --dataset ./data/dataset/val2017.txt \
    --input_size 416
```

**Optimal for FPGA deployment**: 
- `block_size=[4,4]` maps perfectly to 4×4 systolic array PEs on FPGA
- Pruned Conv2D layers can skip entire zero blocks on FPGA (1.8-2.2x speedup)
- INT8 quantization reduces ARM CPU overhead for non-Conv2D layers
- Fine-tuning recovers accuracy lost from pruning
- **Single TFLite model** runs on heterogeneous DE1-SoC (FPGA fabric + ARM CPU)

**Alternative for pure ARM CPU deployment** (no FPGA):
```bash
# Use block_size=[1,16] for ARM NEON SIMD optimization
python prune_model.py \
    --weights ./data/yolov4-tiny.weights \
    --output ./checkpoints/yolov4-tiny-pruned-arm-ft \
    --model yolov4 --tiny \
    --final_sparsity 0.5 \
    --block_size "1,16" \
    --fine_tune True \
    --train_dataset ./data/dataset/train.txt \
    --epochs 2 \
    --batch_size 8 \
    --learning_rate 1e-4
```

## Expected Results

| Configuration | Model Size | Speed | mAP Drop |
|--------------|------------|-------|----------|
| Original FP32 | ~25 MB | 1.0x | Baseline |
| Pruned 30% + INT8 | ~4-5 MB | 2-3x | -1% to -3% |
| Pruned 50% + INT8 | ~3-4 MB | 3-4x | -2% to -5% |
| Pruned 80% + INT8 | ~2-3 MB | 4-5x | -5% to -10% |
| Pruned 50% + gzip | ~1-2 MB | 3-4x | -2% to -5% |

**Notes**:
- Speed improvements depend on hardware sparse operation support
- Model sizes after gzip compression (pruned models compress very well)
- mAP drop can be minimized with fine-tuning
- INT8 quantization provides additional 4x compression beyond pruning

## Implementation Details

### Based on Official TensorFlow Guides

This implementation follows:
- [Pruning with Keras](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras)
- [Comprehensive Pruning Guide](https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide)

### Key Features

1. **Polynomial Decay Schedule**: Gradually increases sparsity from `initial_sparsity` to `final_sparsity`
2. **Magnitude-Based Pruning**: Zeros out weights with smallest absolute values
3. **Layer-wise Control**: Option to prune selectively or skip critical layers
4. **Block Sparsity**: Support for structured sparsity patterns
5. **Fine-Tuning Support**: Optional training to recover accuracy
6. **Proper Export**: Strips pruning wrappers before deployment

### Workflow

```python
# 1. Load pre-trained model
base_model = load_pretrained_model()

# 2. Apply pruning
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
    base_model,
    pruning_schedule=tfmot.sparsity.keras.PolynomialDecay(...)
)

# 3. [Optional] Fine-tune
model_for_pruning.fit(..., callbacks=[
    tfmot.sparsity.keras.UpdatePruningStep(),  # REQUIRED
    tfmot.sparsity.keras.PruningSummaries(...)
])

# 4. Strip pruning wrappers
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

# 5. Save
model_for_export.save(output_path)
```

## Combining with Quantization

For **maximum compression**, combine pruning with INT8 quantization:

```bash
# Step 1: Prune
python prune_model.py \
    --weights ./data/yolov4-tiny.weights \
    --output ./checkpoints/yolov4-tiny-pruned \
    --final_sparsity 0.5

# Step 2: Quantize
python convert_tflite.py \
    --weights ./checkpoints/yolov4-tiny-pruned \
    --output ./checkpoints/yolov4-tiny-pruned-int8.tflite \
    --quantize_mode int8 \
    --dataset ./data/dataset/val2017.txt \
    --num_calibration_images 300

# Step 3: [Optional] Compress with gzip
# The pruned+quantized model compresses very well!
```

**Result**: 5-10x smaller than original with minimal accuracy loss.

## Troubleshooting

### Issue: Sparsity lower than expected

**Solution**: Check that:
- Pruning schedule completed (`end_step` reached)
- `UpdatePruningStep` callback used during training
- Model was stripped with `strip_pruning()` before export

### Issue: Poor accuracy after pruning

**Solutions**:
1. Reduce `final_sparsity` (try 0.3 instead of 0.5)
2. Enable fine-tuning (`--fine_tune=True`)
3. Skip critical layers (`--skip_layers`)
4. Increase `end_step` for more gradual pruning

### Issue: Model not smaller after quantization

**Solutions**:
1. Ensure `strip_pruning()` was called before conversion
2. Apply gzip compression to see full size reduction
3. Verify quantization succeeded with `check_quant.py`

### Issue: "No module named 'tensorflow_model_optimization'"

**Solution**:
```bash
pip install -r requirements.txt
```

The `tensorflow-model-optimization` package is included in `requirements.txt`.

## Best Practices

1. **Start Conservative**: Begin with 30-50% sparsity
2. **Use Fine-Tuning**: Even 1-2 epochs helps recover accuracy
3. **Combine with Quantization**: Pruning + INT8 = best compression
4. **Skip Critical Layers**: Protect first/last layers and detection heads
5. **Use Real Calibration Data**: 300-500 diverse images for quantization
6. **Monitor Per-Layer Sparsity**: Some layers prune better than others
7. **Test on Target Hardware**: Benchmark on actual deployment device
8. **Compress for Distribution**: Always gzip .tflite files

## References

- [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
- [Pruning with Keras Guide](https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras)
- [Comprehensive Pruning Guide](https://www.tensorflow.org/model_optimization/guide/pruning/comprehensive_guide)
- [TFMOT API Docs](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity)
- [TFLite Post-Training Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)

## License

Same as the parent repository.
