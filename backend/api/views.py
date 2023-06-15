import json
from django.forms.models import model_to_dict
from django.http import JsonResponse, HttpResponse
from django.shortcuts import get_object_or_404
from rest_framework import generics
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.conf import settings
import pytesseract
import easyocr
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from django.core.files.storage import FileSystemStorage
from django.utils.datastructures import MultiValueDictKeyError

model = tf.keras.models.load_model(os.getcwd() + '/model_93.h5')
classes_obj = {
    0: 0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:"A", 11:"B", 12:"C", 13:"D", 14:"E", 15:"F", 16:"G", 17:"H",
    18:"I", 19:"J", 20:"K", 21:"L", 22:"M", 23:"N", 24:"O", 25:"P", 26:"Q", 27:"R", 28:"S", 29:"T", 30:"U", 31:"V", 32:"W",
    33:"X", 34:"Y", 35:"Z", 36:"a", 37:"b", 38:"c", 39:"d", 40:"e", 41:"f", 42:"g", 43:"h",
    44:"i", 45:"j", 46:"k", 47:"l", 48:"m", 49:"n", 50:"o", 51:"p", 52:"q", 53:"r", 54:"s", 55:"t", 56:"u", 57:"v", 58:"w",
    59:"x", 60:"y", 61:"z"
}

# Hard coded nutrient benefits
nutrient_object = {
    "gula": {
        "manfaat": ["Membantu mengurangi gejala stres",
                    "Sebagai sumber energi yang cukup cepat memulihkan kondisi tubuh. Zaman dulu, orang banyak membawa gula merah atau gula jawa sebagai bekal perjalanan jauh",
                    "Untuk mencegah pingsan bagi orang yang punya sakit darah rendah"
                   ],
    },
    "minyak": {"manfaat": ["Mendukung Kesehatan Jantung dan Pembuluh Darah",
                           "Bagus untuk Kesehatan Kulit",
                    "Bisa Membantu Menurunkan Berat Badan"
                   ],},
    "garam": {"manfaat": ["Menyeimbangkan kadar elektrolit",
                          "Meringankan gejala fibrosis kistik",
                    "Mencegah hiponatremia"
                   ],},
    "kedelai": {"manfaat": ["Menurunkan tingkat gula pada darah",
                            "Membuat kenyang lebih lama",
                    "Lebih sehat dari daging",
                    "Aman untuk dikonsumsi anak balita"
                   ],}
}
    
def get_bboxes(image_path):

    img = cv2.imread(image_path)
    box_coord = []
    
    reader = easyocr.Reader(['en'])
    txt = reader.readtext(img)

    for t in txt:
        bbox, _, _ = t
        box_coord.append((bbox[0], bbox[2]))
        
    return box_coord
    
def predict_model(IMAGE_PATH):
    img = cv2.imread(IMAGE_PATH)
    results = []
        
    box_coord = get_bboxes(IMAGE_PATH)

     # Iterate every word
    for box in box_coord:
        word = ""
        min_coord, max_coord = box
        xmin, ymin = min_coord
        xmax, ymax = max_coord
        crop_img = img[ymin:ymax, xmin:xmax]
        himg, wimg, _ = crop_img.shape

        tes_box = pytesseract.image_to_boxes(crop_img)
        # Iterate every character
        for bx in tes_box.splitlines():
            bx = bx.split()
            x, y, w, h = int(bx[1]), int(bx[2]), int(bx[3]), int(bx[4])

            crop_char = crop_img[y:h, x:w]
                
            try:
                crop_char = cv2.resize(crop_char, (60,60), interpolation=cv2.INTER_AREA)
            except:
                continue
                
            x = img_to_array(crop_char)
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            classes = model.predict(images, batch_size=10)
            word += str(classes_obj.get(np.argmax(classes)))
                    
        results.append(word)
                
    return " ".join(results)

def show_benefit(material):
    count = 0
    for benefit in nutrient_object[material]['manfaat']:
        count+=1
        print(f"{count}. {benefit}")
    return 

def predict_ocr(IMAGE_PATH):
    img = cv2.imread(IMAGE_PATH)
    reader = easyocr.Reader(['en'])
    txt = reader.readtext(img)
    stop_index = 0
    
    for i in range(len(txt)):
        if txt[i][1].lower() != "komposisi":
            stop_index += 1
            continue
        break
    
    for i in range(stop_index, len(txt)):
        for k, v in nutrient_object.items():
            try:
                if k in txt[i][1].lower():
                    print(f"Benefit dari {k} adalah")
                    show_benefit(k)
                    print()
            except:
                continue
    
    return

# Create your views here.

class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name

@api_view(['GET', 'POST'])
def predict_model(request, *args, **kwargs):
    method = request.method
    fss = CustomFileSystemStorage()

    if method == 'GET':
        return Response({"Status": "OK"})
    
    if method == 'POST':
        try:
            image = request.FILES['image']
            _image = fss.save(image.namem, image)
            path = str(settings.MEDIA_ROOT) + "/" + image.name

            result = predict_model(path)

            return Response({"Result of OCR": f"{result}"})

        except:
            return Response({"Status": "Not OK"})
        


@api_view(['POST'])
def predict_ocr(request, *args, **kwargs):
    method = request.method
    fss = CustomFileSystemStorage()

    if method == 'GET':
        return Response({"Status": "OK"})
    
    if method == 'POST':
        try:
            image = request.FILES['image']
            _image = fss.save(image.namem, image)
            path = str(settings.MEDIA_ROOT) + "/" + image.name

            result = predict_ocr(path)

            return Response({"Result of OCR": f"{result}"})

        except:
            return Response({"Status": "Not OK"})