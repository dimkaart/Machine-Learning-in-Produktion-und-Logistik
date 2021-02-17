import os
import cv2
import random
import numpy as np
import tensorflow as tf
import pytesseract
from core.utils import read_class_names
import core.utils as utils
from core.config import cfg
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import re
from scipy import ndimage 

pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'
# function for cropping each detection and saving as new image
### image_name ggf. wieder entfernen
def crop_objects(img, data, path, allowed_classes, image_name, pixel = 5, license_plate = True):
    boxes, scores, classes, num_objects = data

    if len(img.shape)==3:          # bei farbigem Bild 
        h, w, d = img.shape
    else:                         # Greyscale oder s/w Bild
        h, w = img.shape

    class_names = read_class_names(cfg.YOLO.CLASSES)
    #create dictionary to hold count of objects for image name
    counts = dict()
    if license_plate:
        if num_objects > 0:
            sequence = [0] # only choose one license plate 
        else:
            return
    else:
        box_list = [item[0] for item in boxes[:num_objects]]
        sequence = sorted(range(len(box_list)), key=lambda k: box_list[k])
    for i in sequence:
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            pixel_neu = pixel
            while ymin-pixel_neu < 0 or ymax+pixel_neu > h or xmin-pixel_neu < 0 or xmax+pixel_neu > w:
                pixel_neu = pixel_neu - 1
            cropped_img = img[int(ymin)-pixel_neu:int(ymax)+pixel_neu, int(xmin)-pixel_neu:int(xmax)+pixel_neu]
            # construct image name and join it to path for saving crop properly
            img_name = 'crop_' + str(counts[class_name]) + '_' + image_name + '.png'
            img_path = os.path.join(path, img_name)
            # save image
            cv2.imwrite(img_path, cropped_img)
        else:
            continue
            
def gesamt(image_path, Pfad, save_model, iOU, score, output, pixel, input_size = 416, license_plate = True, **kwargs):
        image_path = os.path.join(Pfad, image_path)

        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.
        
        # get image name by using split method
        image_name = image_path.split('/')[-1]
        image_name = image_name.split('.')[0]
        
        # extract image
        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)


        # apply model
        infer = save_model.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # run non max suppression on detections
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=iOU,
            score_threshold=score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = original_image.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        
        # hold all detection data in one variable
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())
        
        # crop image to size of bounding box with crop_object function
        if license_plate:
            crop_path = os.path.join(os.getcwd(), 'Output', 'Crop')
        else: 
            crop_path = os.path.join(os.getcwd(), 'Output_char', 'Crop', image_name)
        try:
            os.makedirs(crop_path)
        except FileExistsError:
            pass
        if license_plate: 
            crop_objects(cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY), pred_bbox, crop_path, allowed_classes, image_name, pixel = pixel)
        else:
            crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY), pred_bbox, crop_path, allowed_classes, image_name, license_plate = False, pixel = pixel)

        # initialize final image
        if license_plate:
            image = utils.draw_bbox(original_image, pred_bbox,  allowed_classes=allowed_classes)
        else:
            image = utils.draw_bbox(original_image, pred_bbox,  allowed_classes=allowed_classes, show_label = False)
        
        image = Image.fromarray(image.astype(np.uint8))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

        # save final image
        cv2.imwrite(output + 'Erkennung_' + image_name + '.png', image)
        
# OCR function for each character
def ocr_neu(file, root, subdirectory, tune = False):
    img = cv2.imread(os.path.join(os.path.join(root, subdirectory),file))
    img = cv2.resize(img, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    if tune: # tune parameters oem and psm
        config_tune = '--oem ' + kwargs['oem'] +' -l deu --psm ' + kwargs['psm'] + ' -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '
        data = pytesseract.image_to_string(blur, config=config_tune)
    else: # result after tuning 
        data = pytesseract.image_to_string(blur, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -l deu --psm 13 --oem 1')
    data = re.sub('[\W_]+', '', data)
    return(data)

# rotation function
def rotation(Pfad, image_path, lower, upper):
    image = cv2.imread(Pfad+image_path)
    # create border
    row, col = image.shape[:2]
    bottom = image[0:row, 0:col]
    mean = cv2.mean(bottom)[0]
    bordersize = 5
    border = cv2.copyMakeBorder(
        image,
        top=bordersize,
        bottom=bordersize,
        left=bordersize,
        right=bordersize,
        borderType=cv2.BORDER_CONSTANT,
        value=[mean, mean, mean]
        )
    ## edge detection on image with border
    gray = cv2.cvtColor(border, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # find angle
    cnts, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contour with the largest area is possibly the plate
    max_area = 0
    max_cnt = None
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if(area > max_area):
            max_area = area
            max_cnt = cnt
    if max_cnt is not None:
        min_rect = cv2.minAreaRect(max_cnt)
        (midpoint, widthheight, angle) = min_rect
        # Get the image size
        # NumPy stores image matricies backwards
        image_size = (border.shape[1], border.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack([cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])
        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])
        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5
        # Obtain the rotated coordinates of the image corners
        rotated_coords = [(np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],(np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],(np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],(np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # translation matrix to keep the image centred
        trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)],[0, 1, int(new_h * 0.5 - image_h2)],[0, 0, 1]])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform --> (-0° to -45° License Plates, which are tilted to the left and -46° and -90° License Plates, which are tilted to the right)
        rotation1 = cv2.warpAffine(image,affine_mat,(new_w, new_h),flags=cv2.INTER_LINEAR)
        # correction of rotation for -45 > angle >= -90
        rotation2 = ndimage.rotate(rotation1, 90)
        # if angle is not between -20° and -70°, keep original image in pipeline
        if (lower >= angle >= -45.0):
            rotated_image = rotation1
            rotated_image = Image.fromarray(rotated_image)
            rotated_image = rotated_image.convert('L')
            rotated_image.save(Pfad+image_path)
        elif (-45 > angle >= upper):
            rotated_image = rotation2
            rotated_image = Image.fromarray(rotated_image)
            rotated_image = rotated_image.convert('L')
            rotated_image.save(Pfad+image_path)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(Pfad+image_path, image)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(Pfad+image_path, image)
       
# OCR function for whole license plate (if no character is recognized)
def ocr_tesseract(image_path, Pfad, **kwargs):
    original_image = cv2.imread(os.path.join(Pfad, image_path))
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    blur = cv2.medianBlur(thresh, 3)
    blur = cv2.resize(blur, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(blur, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -l deu --psm 13 --oem 1')
    text = re.sub('[\W_]+', '', text)   
    return(text)

