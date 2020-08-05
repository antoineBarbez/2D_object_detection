import tensorflow as tf


def get_feature_extractor_model(image_shape):
    i = tf.keras.Input(image_shape, dtype=tf.uint8)
    x = tf.cast(i, tf.float32)
    x = tf.keras.applications.resnet.preprocess_input(x)
    resnet_50 = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=image_shape)
    core = tf.keras.Model(inputs=resnet_50.input, outputs=resnet_50.get_layer("conv4_block6_out").output)
    x = core(x)
    return tf.keras.Model(inputs=[i], outputs=[x])
