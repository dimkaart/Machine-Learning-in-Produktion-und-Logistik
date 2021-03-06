# Define environment
import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS, FlagValues
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import Levenshtein as ls
import shutil
import itertools 
import pandas as pd 
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

# delete old detection results
print('\nEntferne alte Ergebnisse...\n')
for filename in os.listdir("./Output/Crop/"):
   os.remove("./Output/Crop/"+filename)
for filename in os.listdir("./Output/Erkennung/"):
   os.remove("./Output/Erkennung/"+filename)
for folder in os.listdir("./Output_char/Crop/"):
   shutil.rmtree("./Output_char/Crop/"+folder)
for filename in os.listdir("./Output_char/Erkennung/"):
   os.remove("./Output_char/Erkennung/"+filename)
os.remove("./Output/Ergebnis.txt")

# initialize solution .txt file
print('Bereite Ergebnisdatei vor...\n')
ErgFile=open("./Output/Ergebnis.txt","w")

# set up
Output='./Output/Erkennung/'
Output_char = './Output_char/Erkennung/'
Model='./checkpoints/custom-416'
Model_char = './checkpoints/char-416'

# thresholds to accept found bounding box as license plate
iOU=0.45
score=0.50

# flags for outer functions 
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')

# main function
def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    cfg.YOLO.CLASSES = "./data/classes/custom.names"
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416 #Resize image
    
    # load model
    print('Lade erstes YOLOv4 Modell...\n')
    saved_model_loaded = tf.saved_model.load(Model, tags=[tag_constants.SERVING])

    # loop through images in folder and run Yolov4 model on each
    Pfad='./Input/'
    input_names = []
    input_vergleich = []
    print('Betrachte Bilder im Input Ordner...')
    for image_path in os.listdir(Pfad):
        input_vergleich += [image_path.split('.')[0]]
        print('Aktuelles Auto: ' + image_path[:-4]+ '...')
        input_crop = image_path.split('_')[0]
        input_crop = input_crop.split('.')[0]
        input_names += [input_crop]
        gesamt(Pfad = Pfad, image_path = image_path, save_model = saved_model_loaded, 
            iOU = iOU, score = score, output = Output, pixel = 5) 
        # functions.py applies YOLOv4 on the image to find the license plate and to crop along the result

    Pfad = './Output/Crop//'
    if rotate: #rotation
        for image_path in os.listdir(Pfad):
            rotation(Pfad = Pfad, image_path= image_path, lower = -20, upper = -70)
    cfg.YOLO.CLASSES = "./data/classes/char.names"
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    # load model
    print('\nLade zweites YOLOv4 Modell...\n')
    saved_model_char = tf.saved_model.load(Model_char, tags=[tag_constants.SERVING])
    Pfad = './Output/Crop/'
    print('\nBetrachte Kennzeichen...')
    if not tune: # after tuning
        Pfad = './Output/Crop//'
        output_names = []
        output_vergleich = []
        for image_path in os.listdir(Pfad):
            print('Aktuelles Kennzeichen: ' + image_path[:-4]+ '...')
            vergleich = image_path.split('.png')[0]
            vergleich = vergleich.split('crop_1_')
            output_vergleich += [vergleich[1]]
            name_neu = image_path.split('_')[2]
            name_neu = name_neu.split('.')[0]
            output_names += [name_neu]
            gesamt(Pfad = Pfad, image_path = image_path, save_model = saved_model_char, 
                    iOU = iOU, score = 0.5, output=Output_char, license_plate = False, pixel = 2) # apply Yolov4 on the image to find characters 
        distance_index = []
        for i in input_names: # find images where no license plate has been found
            if (i in output_names):
                distance_index += [1]
            else:
                distance_index += [0]
        list_0 = [i for i, value in enumerate(distance_index) if value == 0]
        list_1 = [i for i, value in enumerate(distance_index) if value == 1]
        distance = [None] * len(input_names) # distance list
        for i in list_0:
            distance[i] = len(input_names[i]) # if no license found => distance = number of character of true recognition
        insgesamt = []         
        path = './Output_char/Crop//'
        print('\nBeginne OCR...\n')
        for root, subdirectories, files in os.walk(path): # loop through every license plate and detected characters
            for subdirectory in subdirectories:
                result = ''
                original = subdirectory.split('_')[2]
                j = 0
                for file in os.listdir(os.path.join(root, subdirectory)): # do ocr on all characters
                    result_neu = ocr_neu(file, root, subdirectory)
                    if crop:
                        if len(result_neu) > 1:
                            result_neu = ocr_neu(file, root, subdirectory, resize = True)
                            if len(result_neu) > 1:
                                j += 1
                    result += result_neu
                if result == '': # if no character has been recognized => use tesseract on whole license plate
                    erg = ocr_tesseract(image_path = image_path, Pfad = Pfad)
                    result = erg
                insgesamt += [result]
                if not final: # outside final challenge calculate distance
                    index_name = input_names.index(original)
                    distance[index_name] = ls.distance(result, original)
        print('Bereite die Ausgabe vor...\n')
        print('erkannte Kennzeichen:', insgesamt)
        if not final:
            if distance:
                print('Distanz:', distance)
                print('durchschnittliche Levenshtein-Distanz:', np.mean(distance))
        print('Bilder in Input:', input_vergleich)
        print('Bilder in Output:', output_vergleich)
        print('\nSchreibe Ergebnisdatei...\n')
        for index in input_vergleich: # create txt output file
            if not output_vergleich:
                ErgFile.write(index + ":" + '\n')
            elif (index in output_vergleich):
                stelle = output_vergleich.index(index)
                ErgFile.write(index + ":" + insgesamt[stelle] + '\n')
            else: 
                ErgFile.write(index + ":" + '\n')
        ErgFile.close()
        print('Fertig!')
    else: # tuning
        Pfad = './Output/Crop//'
        distance_tess = []
        for image_path in os.listdir(Pfad): # do ocr on whole license plate
            name_neu = image_path.split('_')[2]
            name_neu = name_neu.split('.')[0]
            res = ocr_tesseract(image_path = image_path, Pfad = Pfad)
            distance_tess += [ls.distance(res, name_neu)]
        print('tess_distance:', np.mean(distance_tess))
        evaluation = []
        params = {'psm': ['10', '13'], 'oem': ['1'], 
            'pixel': [x for x in range(0, 4)], 'score': [round(x, 2) for x in np.arange(0.5, 0.76, 0.1)]} # possible parameters
        keys = list(params)
        for values in itertools.product(*map(params.get, keys)):
            for folder in os.listdir("./Output_char/Crop/"):
                shutil.rmtree("./Output_char/Crop/"+folder)
            for filename in os.listdir("./Output_char/Erkennung/"):
                os.remove("./Output_char/Erkennung/"+filename)
            print('values:', values)
            name_crop = []
            for image_path in os.listdir(Pfad):
                name_neu = image_path.split('_')[2]
                name_neu = name_neu.split('.')[0]
                name_crop += [name_neu]
                gesamt(Pfad = Pfad, image_path = image_path, save_model = saved_model_char, 
                        iOU = iOU, output=Output_char, license_plate = False, **dict(zip(keys, values))) 
            insgesamt = []
            distance = []
            path = './Output_char/Crop//'
            for root, subdirectories, files in os.walk(path):
                for subdirectory in subdirectories:
                    result = ''
                    original = subdirectory.split('_')[2]
                    print("original:", original)
                    for file in os.listdir(os.path.join(root, subdirectory)):
                        result += ocr_neu(file, root, subdirectory, tune = True, **dict(zip(keys, values)))
                    insgesamt += [result]
                    distance += [ls.distance(result, original)]
            print('erkanntes Kennzeichen:', insgesamt)
            print('Distanz:', distance)
            evaluation.append([values[0], values[1], values[2], values[3], np.mean(distance)])
        evaluation_df = pd.DataFrame(evaluation, columns = ['oem', 'psm', 'pixel', 'score', 'mean(distance)']) # create table as output
        evaluation_sort = evaluation_df.sort_values(by =['mean(distance)'])
        fig, ax =plt.subplots(figsize=(12,4))
        ax.axis('tight')
        ax.axis('off')
        the_table = ax.table(cellText=evaluation_sort.values,colLabels=evaluation_sort.columns,loc='center')
        pp = PdfPages("Evaluation.pdf")
        pp.savefig(fig, bbox_inches='tight')
        pp.close()     

if __name__ == '__main__':
    try:
        rotate = False
        tune = False
        final = True
        app.run(main)
    except SystemExit:
        pass
