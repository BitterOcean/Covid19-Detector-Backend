from .serializers import PostSerializer
from .models import Post
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('./model/model.h5')


def predict(img_name):
    img = tf.keras.preprocessing.image.load_img(img_name, grayscale=False, target_size=(64, 64))
    disease_class = ['Covid-19', 'Non Covid-19']
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    custom = model.predict(x)
    p = custom[0]
    ind = np.argmax(p)
    return disease_class[ind], p


class PostView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @staticmethod
    def get(request, *args, **kwargs):
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)

    @staticmethod
    def post(request, *args, **kwargs):
        posts_serializer = PostSerializer(data=request.data)
        if posts_serializer.is_valid():
            posts_serializer.save()
            result, probability = predict('.' + posts_serializer.data.get('image'))
            response_data = posts_serializer.data
            response_data.update({'prediction': result, 'probability': probability})
            return Response(response_data, status=status.HTTP_201_CREATED)
        else:
            print('error', posts_serializer.errors)
            return Response(posts_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
