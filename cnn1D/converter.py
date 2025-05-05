import tensorflow as tf

model = tf.keras.models.load_model('./model/cnn1d_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open("cnn1d_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ 변환 완료!")
