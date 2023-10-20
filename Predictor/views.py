from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import io

model = load_model('./model/model.h5')

def predict(img_name):
    disease_class = ['Covid-19', 'Non Covid-19']
    x = image.img_to_array(img_name)
    x = np.expand_dims(x, axis=0)
    x /= 255
    custom = model.predict(x)
    p = custom[0]
    ind = np.argmax(p)
    return disease_class[ind], p

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
