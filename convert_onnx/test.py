import tensorflow as tf
import tf2onnx
from tensorflow.keras.models import load_model

# Load model
model = load_model("numbers_model.h5", compile=False)

# input signature (عدل الأبعاد إذا لزم)
spec = (tf.TensorSpec((None, 90, 42), tf.float32, name="input"),)

@tf.function(input_signature=spec)
def model_fn(x):
    return model(x)

# Convert using the tf.function (NOT a concrete function)
tf2onnx.convert.from_function(
    model_fn,
    input_signature=spec,
    opset=15,
    output_path="numbers_model.onnx"
)

print("✅ Saved numbers_model.onnx")
