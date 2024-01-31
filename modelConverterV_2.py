import tensorflow as tf
import tf2onnx
from tf2onnx import convert
from tf2onnx.convert import from_keras
import tflite2onnx

def tf_tflite16(path):
    h5_model_path = path 

    model = tf.keras.models.load_model(h5_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    global tflite16_model_path
    tflite16_model_path = 'models/VGG19_float16.tflite'
    with open(tflite16_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f'TFLite model with float16 precision saved to: {tflite16_model_path}')

def tflite16_onnx():
    tflite_model_path = "models/VGG19_float16.tflite"
    onnx_path = 'models/VGG19_float16_onnx.onnx'

    tflite2onnx.convert(tflite_model_path, onnx_path)

def tf_onnx(path):
    h5_model_path = path
    model = tf.keras.models.load_model(h5_model_path)
    onnx_model, _ = tf2onnx.convert.from_keras(model)

    onnx_model_path = 'models/efficientnet_32_onnx.onnx'
    with open(onnx_model_path, 'wb') as f:
        f.write(onnx_model.SerializeToString())

    print(f'ONNX model saved to: {onnx_model_path}')