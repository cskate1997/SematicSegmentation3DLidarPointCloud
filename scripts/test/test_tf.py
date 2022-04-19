import tensorflow as tf

print("GPUs: ", tf.test.is_gpu_available())
# print(tf.test.gpu_device_name())
# print(tf.keras.backend.get_session().list_devices())
i = tf.keras.backend.get_session().list_devices()
for x in i:
    print(x)
    print(x.name)
    print(type(x))
    print(x.device_type)
j = [x for x in i if 'device:gpu' in x.name.lower()]
print(j)
# for i in tf.keras.backend.get_session().list_devices():
#     print(i)
#     print()