from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import io

# model = load_model('./model/model.h5')

# def predict(img_name):
#     disease_class = ['Covid-19', 'Non Covid-19']
#     x = image.img_to_array(img_name)
#     x = np.expand_dims(x, axis=0)
#     x /= 255
#     custom = model.predict(x)
#     p = custom[0]
#     ind = np.argmax(p)
#     return disease_class[ind], p


interpreter = tf.lite.Interpreter(model_path='./model/model.tflite')
interpreter.allocate_tensors()

# Define the prediction function
def predict(image_file):
    img = image.load_img(image_file, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    return prediction[0]


@csrf_exempt
def predict_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        img_content = image_file.read()
        img = image.load_img(io.BytesIO(img_content), grayscale=False, target_size=(64, 64))
        result, probability = predict(img)
        response = {
            'prediction': result,
            'probability': probability.tolist()
        }
        return JsonResponse(response)
    else:
        return JsonResponse({'error': 'Invalid request method.'})
