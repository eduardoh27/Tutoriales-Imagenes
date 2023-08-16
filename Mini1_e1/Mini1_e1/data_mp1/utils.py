import json
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_annotations(fold, img_name, annotations_json_name='_annotations.coco.json', interest_class=-1):
    '''
    Parameters
    ----------
    fold : 'train', 'valid' or 'test' depending on the source fold the image comes from.
    img_name : image of interest from the 'fold' folder name (including the extension).
    annotations_json_name : name of the json located into the 'fold' folder.
    interest_class : Class of interest label to draw only those annotations. If -1 all the annotations are shown.

    Returns
    -------
    img : img_name image content overwritten with the corresponding annotations.

    '''
    # Classes of the Aquarium Dataset
    CLASSES = ['cells', 'platelet', 'red blood cell', 'white blood cell',]

    # Colors for visualization (one per class)
    COLORS = [[0,0,0], [0, 255, 0], [255, 0, 0], [0, 0, 255]]
    
    # Obtaining the images and annotations from de json file
    json_data = json.load(open(os.path.join('data_mp1','BCCD', fold, annotations_json_name), 'r'))
    images = json_data['images']
    annotations = json_data['annotations']
    
    # Importing the image of interest
    img = plt.imread(os.path.join('data_mp1','BCCD', fold, img_name))
    for i in images:
        # Selection of the id of the image of interest
        if i['file_name'] == img_name:
            train_id = i['id']
            
            # Variables where each annotation, its class type and repsective color to use are added.
            boxes = []
            classes = []
            colors = []
            
            for i in annotations:
                # Adding all the annotations that correspond to the desired image.
                if i['image_id'] == train_id and interest_class != -1:
                    if i['category_id'] == interest_class:
                        boxes.append(i['bbox'])
                        classes.append(CLASSES[i['category_id']])
                        colors.append(COLORS[i['category_id']])
                elif i['image_id'] == train_id:
                    boxes.append(i['bbox'])
                    classes.append(CLASSES[i['category_id']])
                    colors.append(COLORS[i['category_id']])
            # Visualization of the annotations
            size=2
            for cl, (xmin, ymin, width, height), c in zip(classes, boxes, colors):
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmin+width), int(ymin+height)), c, size)
                text = f'{cl}'
                cv2.putText(img=img, text=text, org=(xmin, ymin-5), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1.2, color=c,thickness=size-1)

    return img
        
def pred_score(img):
    '''
    Parameters
    ----------
    img : Cropted 3 channel image to be scored.

    Returns
    -------
    score : Score asociated with the input image.

    '''
    # Histograms of each channer are calculated separately
    h_r = np.histogram(img[:, :, 0], 25, [0, 255])[0]
    h_g = np.histogram(img[:, :, 1], 25, [0, 255])[0]
    h_b = np.histogram(img[:, :, 2], 25, [0, 255])[0]
    # All histograms are concatenated in one single array
    hist = np.concatenate((h_r, h_g, h_b))
    # The concatenated color histogram is normalized
    hist = hist / np.sum(hist)
    # A pre-selected histogram that is going to be used for comparison is defined
    ref_hist = [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     6.71609714e-05, 4.82215775e-03, 1.80931657e-02, 3.09209112e-02, 2.17198582e-02,
     1.00338491e-02, 3.35267569e-02, 9.43477326e-02, 5.47093273e-02, 4.83558994e-02,
     1.65081668e-02, 2.28347303e-04, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     0.00000000e+00, 6.17880937e-04, 9.63088330e-03, 2.51316355e-02, 2.84493875e-02,
     1.16725768e-02, 6.62207178e-03, 6.29969912e-03, 1.70320224e-02, 6.57102944e-02,
     5.43197937e-02, 2.33585859e-02, 1.15113905e-02, 2.51047711e-02, 3.57833656e-02,
     1.20889749e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
     0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.35665162e-03,
     1.22367290e-02, 1.66290565e-02, 3.10418010e-02, 7.99349882e-02, 9.17687513e-02,
     5.80002149e-02, 3.13641737e-02, 9.16075650e-03, 1.71932087e-03, 1.20889749e-04]
    # The distance is calculated using the intersection distance
    score = cv2.compareHist(np.array(ref_hist,dtype=np.float32),np.array(hist,dtype=np.float32),2)
    return score


def detections(conf_thresh, jaccard_thresh, anot_file, pred_file):
    def intersection(coords1, coords2):
        # Se calculan las coordenadas del primer recuadro
        x11 = np.round(coords1[1], 0)
        x12 = np.round(coords1[1] + coords1[3], 0)
        y11 = np.round(coords1[0], 0)
        y12 = np.round(coords1[0] + coords1[2], 0)

        # Se calculan las coordenadas del segundo recuadro
        x21 = np.round(coords2[1], 0)
        x22 = np.round(coords2[1] + coords2[3], 0)
        y21 = np.round(coords2[0], 0)
        y22 = np.round(coords2[0] + coords2[2], 0)

        # si no se intersectan se da 0
        if x12 < x21 or y12 < y21 or x11 > x22 or y22 < y11:
            return 0
        else:
            # si se intersectan se calcula el area del recuadro intermedio
            xs = np.sort([x11, x21, x22, x12])
            ys = np.sort([y11, y21, y22, y12])
            inter = (xs[2] - xs[1]) * (ys[2] - ys[1])
            union = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - inter
            return inter / union

    # Se abren los archivos
    file1 = open(anot_file)
    data1 = json.load(file1)
    anotations_temp = data1['annotations']
    anotations = []
    for anotation in anotations_temp:
        if anotation['category_id'] == 3:
            anotations.append(anotation)

    file2 = open(pred_file)
    predictions = json.load(file2)

    # Inicio las variables
    TP = 0
    FP = 0
    FN = 0

    # Se extraen las imagenes
    images = data1['images']

    # Se recorren las imagenes
    for image in images:

        # se extrae el id
        id = image['id']

        # Se incluyen unicamente las anotaciones y predicciones de la imagen
        anots = []
        preds = []
        for pred in predictions:
            if pred['image_id'] == id:
                preds.append(pred)
        for anot in anotations:
            if anot['image_id'] == id:
                anots.append(anot)
        # se recorre cada anotación
        for anot in anots:
            # se asume que la anotacion no fue encontrada
            anot_found = False
            # se recorren las predicciones
            for pred in preds:
                # se revisa si la prediccion supera el umbral de confianza
                if pred['score'] >= conf_thresh:
                    # Se calcula la intersección
                    intersect = intersection(anot['bbox'], pred['bbox'])
                    # se verifica si se supera el umbral de jaccard
                    if intersect >= jaccard_thresh:
                        preds.remove(pred)
                        TP += 1
                        anot_found = True
                else:
                    # se quitan las predicciones que no superen el umbral
                    preds.remove(pred)
            # si al final del recorrido no se tiene predicción se considera falso negativo
            if not anot_found:
                FN += 1
        # se suman todas las predicciones sobrantes
        FP += len(preds)
    return TP, FP, FN