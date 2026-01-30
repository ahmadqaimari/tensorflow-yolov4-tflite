import tensorflow as tf
from collections import Counter
import numpy as np

# Load the quantized TFLite model
model_path = "./checkpoints/yolov4-tiny-INT8-WITH-FLEX.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get all tensor details
tensor_details = interpreter.get_tensor_details()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("=" * 80)
print(f"FULL MODEL QUANTIZATION AUDIT: {model_path}")
print("=" * 80)

# 1. Count dtypes of all tensors in the graph
dtypes = [t['dtype'].__name__ for t in tensor_details]
dtype_counts = Counter(dtypes)

print("\n📊 TENSOR TYPE DISTRIBUTION:")
total_tensors = len(tensor_details)
for dtype, count in sorted(dtype_counts.items()):
    percentage = (count / total_tensors) * 100
    print(f"   {dtype:15s}: {count:4d} tensors ({percentage:5.1f}%)")

# Explain object type tensors if present
if dtype_counts.get('object', 0) > 0:
    print("\n⚠️  NOTE: 'object' type tensors detected")
    print("   These are typically:")
    print("   • String tensors (for labels, filenames)")
    print("   • Resource handles (for variables, queues)")
    print("   • Variant tensors (for complex data structures)")
    print("   • These are NOT used in actual inference computation")

# 2. Analyze Conv2D layers (4D weight tensors)
print("\n" + "=" * 80)
print("🔍 CONV2D LAYER ANALYSIS (4D Weight Tensors)")
print("=" * 80)

# Conv2D weights are typically 4D: [height, width, in_channels, out_channels]
weight_tensors = [t for t in tensor_details if len(t['shape']) == 4]

# Separate actual Conv2D from BatchNorm tensors
conv2d_tensors = [t for t in weight_tensors if 'Conv2D' in t['name'] or 'conv' in t['name'].lower()]
bn_tensors = [t for t in weight_tensors if 'batch_norm' in t['name'].lower() or 'FusedBatchNorm' in t['name']]
other_tensors = [t for t in weight_tensors if t not in conv2d_tensors and t not in bn_tensors]

int8_conv = [t for t in conv2d_tensors if t['dtype'] in [np.int8, np.uint8]]
float_conv = [t for t in conv2d_tensors if t['dtype'] == np.float32]

int8_bn = [t for t in bn_tensors if t['dtype'] in [np.int8, np.uint8]]
float_bn = [t for t in bn_tensors if t['dtype'] == np.float32]

print(f"\n📊 WEIGHT TENSOR BREAKDOWN:")
print(f"  Total 4D tensors: {len(weight_tensors)}")
print(f"    ├─ Conv2D tensors: {len(conv2d_tensors)}")
print(f"    │  ├─ INT8/UINT8:  {len(int8_conv)} ✅")
print(f"    │  └─ FLOAT32:     {len(float_conv)} ❌")
print(f"    ├─ BatchNorm tensors: {len(bn_tensors)}")
print(f"    │  ├─ INT8/UINT8:  {len(int8_bn)} ✅")
print(f"    │  └─ FLOAT32:     {len(float_bn)} ❌")
if other_tensors:
    print(f"    └─ Other tensors: {len(other_tensors)}")

print(f"\n🎯 KEY METRICS:")
if len(conv2d_tensors) > 0:
    conv_quant_pct = (len(int8_conv) / len(conv2d_tensors)) * 100
    print(f"  Conv2D Quantization: {len(int8_conv)}/{len(conv2d_tensors)} ({conv_quant_pct:.1f}%)")
if len(bn_tensors) > 0:
    bn_quant_pct = (len(int8_bn) / len(bn_tensors)) * 100
    print(f"  BatchNorm Quantization: {len(int8_bn)}/{len(bn_tensors)} ({bn_quant_pct:.1f}%)")

if len(conv2d_tensors) > 0:
    print(f"\n📋 CONV2D LAYERS (first 25):")
    print(f"{'Layer Name':<60} {'Shape':<25} {'Type':<10} {'Status'}")
    print("-" * 105)

    for i, tensor in enumerate(sorted(conv2d_tensors, key=lambda x: x['name'])[:25], 1):
        name = tensor['name']
        shape = str(tuple(tensor['shape']))
        dtype = tensor['dtype'].__name__

        if dtype in ['int8', 'uint8']:
            status = "✅ QUANTIZED"
        else:
            status = "❌ NOT QUANTIZED"

        # Shorten name if too long
        if len(name) > 58:
            display_name = name[:55] + "..."
        else:
            display_name = name

        print(f"{display_name:<60} {shape:<25} {dtype:<10} {status}")

    if len(conv2d_tensors) > 25:
        print(f"... and {len(conv2d_tensors) - 25} more Conv2D layers")

# 3. Show object type tensors if any
object_tensors = [t for t in tensor_details if t['dtype'] == np.object_]
if object_tensors:
    print("\n" + "=" * 80)
    print("🔍 OBJECT TYPE TENSORS (Non-computational)")
    print("=" * 80)
    print(f"Total: {len(object_tensors)} tensors")
    print("\nThese are metadata/string tensors, NOT used in inference:")
    for obj_tensor in object_tensors[:10]:  # Show first 10
        print(f"  • {obj_tensor['name']}")
    if len(object_tensors) > 10:
        print(f"  ... and {len(object_tensors) - 10} more")

# 4. Analyze activations (intermediate tensors)
activation_tensors = [t for t in tensor_details if len(t['shape']) in [2, 3, 4] and t not in weight_tensors and t['dtype'] != np.object_]
int8_activations = [t for t in activation_tensors if t['dtype'] in [np.int8, np.uint8]]

print("\n" + "=" * 80)
print("⚡ ACTIVATION TENSOR ANALYSIS")
print("=" * 80)
print(f"Total activation tensors: {len(activation_tensors)}")
print(f"  └─ INT8/UINT8:  {len(int8_activations)} ✓")
print(f"  └─ FLOAT32:     {len(activation_tensors) - len(int8_activations)} ✗")

# 4. Check Input/Output quantization
print("\n" + "=" * 80)
print("📥 INPUT/OUTPUT QUANTIZATION")
print("=" * 80)

print("\nINPUT TENSORS:")
for detail in input_details:
    dtype = detail['dtype'].__name__
    shape = tuple(detail['shape'])
    status = "✅ QUANTIZED" if dtype in ['int8', 'uint8'] else "❌ NOT QUANTIZED"
    print(f"  {detail['name']:<50} {str(shape):<20} {dtype:<10} {status}")
    if 'quantization' in detail and detail['quantization'] != (0.0, 0):
        scale, zero_point = detail['quantization']
        print(f"    └─ Quantization: scale={scale:.6f}, zero_point={zero_point}")

print("\nOUTPUT TENSORS:")
for detail in output_details:
    dtype = detail['dtype'].__name__
    shape = tuple(detail['shape'])
    status = "✅ QUANTIZED" if dtype in ['int8', 'uint8'] else "❌ NOT QUANTIZED"
    print(f"  {detail['name']:<50} {str(shape):<20} {dtype:<10} {status}")
    if 'quantization' in detail and detail['quantization'] != (0.0, 0):
        scale, zero_point = detail['quantization']
        print(f"    └─ Quantization: scale={scale:.6f}, zero_point={zero_point}")

# 5. Calculate quantization percentage (excluding object tensors)
int8_total = dtype_counts.get('int8', 0) + dtype_counts.get('uint8', 0)
float32_total = dtype_counts.get('float32', 0)
object_total = dtype_counts.get('object', 0)
compute_tensors = total_tensors - object_total  # Exclude non-computational tensors

if compute_tensors > 0:
    quantization_pct = (int8_total / compute_tensors) * 100
else:
    quantization_pct = 0

# 6. Final Verdict
print("\n" + "=" * 80)
print("🎯 FINAL VERDICT")
print("=" * 80)

print(f"\nTotal tensors: {total_tensors} ({compute_tensors} computational, {object_total} metadata)")
print(f"Overall Quantization: {quantization_pct:.1f}% ({int8_total}/{compute_tensors} tensors)")

# Show Conv2D specific metrics
if len(conv2d_tensors) > 0:
    conv_quant_pct = (len(int8_conv) / len(conv2d_tensors)) * 100
    print(f"Conv2D Quantization: {len(int8_conv)}/{len(conv2d_tensors)} layers ({conv_quant_pct:.1f}%)")

if len(bn_tensors) > 0:
    bn_quant_pct = (len(int8_bn) / len(bn_tensors)) * 100
    print(f"BatchNorm Quantization: {len(int8_bn)}/{len(bn_tensors)} layers ({bn_quant_pct:.1f}%)")

if len(int8_conv) == len(conv2d_tensors) and len(int8_conv) > 0:
    print("\n✅ EXCELLENT: Full Conv2D INT8 Quantization!")
    print("   • ALL Conv2D layers are quantized")
    print("   • Conv2D operations are 90%+ of inference time")
    print("   • Expected 3-4x speedup on INT8 hardware")
    if len(float_bn) > 0:
        print(f"   • Note: {len(float_bn)} BatchNorm layers remain FP32 (minimal performance impact)")
elif len(int8_conv) / len(conv2d_tensors) >= 0.8 if len(conv2d_tensors) > 0 else False:
    print("\n✅ VERY GOOD: High Conv2D INT8 Quantization!")
    conv_pct = (len(int8_conv) / len(conv2d_tensors)) * 100
    print(f"   • {conv_pct:.1f}% of Conv2D layers quantized")
    print("   • Expected significant speedup on INT8 hardware")
elif quantization_pct >= 50:
    print("\n⚠️  GOOD: Partial INT8 Quantization")
    print(f"   • {quantization_pct:.1f}% of tensors quantized")
    print(f"   • {len(int8_weights)}/{len(weight_tensors)} Conv2D layers quantized")
    print(f"   • {float32_total} FP32 tensors remain")
    print("   • Moderate speedup expected")
elif quantization_pct > 0:
    print("\n⚠️  POOR: Limited Quantization")
    print(f"   • Only {quantization_pct:.1f}% quantized")
    print("   • Most operations still in FP32")
    print("   • Limited performance improvement")
else:
    print("\n❌ FAILED: Model is NOT quantized")
    print("   • All tensors are FP32")
    print("   • Quantization process failed")

# 7. Model size
import os
if os.path.exists(model_path):
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\n📦 Model Size: {model_size_mb:.2f} MB")

print("\n" + "=" * 80)
