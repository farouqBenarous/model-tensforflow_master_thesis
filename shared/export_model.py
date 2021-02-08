import tensorflow as tf
import  sys

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
model_path = sys.argv[1]
model_export_path = sys.argv[2]

if not model_path :
    print("mode given : ",model_path," not found")
    exit(0)

if not model_export_path :
    print("mode export path : ",model_export_path," not found")
    exit(0)

model = tf.keras.models.load_model(model_path)
export_path = model_export_path

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})
