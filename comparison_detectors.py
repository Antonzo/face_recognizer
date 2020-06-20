import cv2
import os
from math import pi
from fnmatch import fnmatch
from detectors import DetectorSSD, DetectorVJ, DetectorHOG, DetectorMMOD, DetectorLBP
from datetime import datetime

# пути к БД с изображениями и аннотациями
args = {}
args['ann'] = 'FDDB-folds'
args['pics'] = 'originalPics'

# датасет в необходимом формате
dataset = {}
dataset['images'] = []
dataset['annotations'] = []

# инициализация всех детекторов
detector_ssd = DetectorSSD()
detector_vj = DetectorVJ()
detector_hog = DetectorHOG()
detector_mmod = DetectorMMOD()
detector_lbp = DetectorLBP()
detectors = [detector_ssd, detector_vj, detector_hog, detector_mmod, detector_lbp]


all_count = 0 # всего лиц
iou_75 = 0 # c IOU >= 75
iou_50 = 0 # c IOU >= 50
iou_avg = 0 # средний IOU
false_25 = 0 # c IOU <= 50
non_det = 0 # нераспознанных
total_time = 0 # затраченное время

# добавляет изображение к датасету (id, путь до файла)
def add_image(image_path):

    image_id = len(dataset['images'])
    dataset['images'].append(
    {'id': image_id,
     'file_name': image_path})
    return image_id


# добавляет аннотацию к датасету (id лица, id изображения, координаты ROI лица)
def add_bbox(image_id, start_x, start_y, end_x, end_y):

    dataset['annotations'].append(
    {'id': len(dataset['annotations']),
     'image_id': int(image_id),
     'bbox': [start_x, start_y, end_x, end_y]})
 
    
# трансформация эллипса в прямоугольник
def ellipse_to_rect(params):
    rad_x = params[0]
    rad_y = params[1]
    angle = params[2] * 180.0 / pi
    center_x = params[3]
    center_y = params[4]
    pts = cv2.ellipse2Poly((int(center_x), int(center_y)), (int(rad_x), int(rad_y)),
                          int(angle), 0, 360, 10)
    rect = cv2.boundingRect(pts)
    left = rect[0]
    top = rect[1]
    right = rect[0] + rect[2]
    bottom = rect[1] + rect[3]
    return left, top, right, bottom


# считывание FDDB датасета и формирование словаря dataset
def fddb_dataset(annotations, images):

    for d in os.listdir(annotations):
        if fnmatch(d, 'FDDB-fold-*-ellipseList.txt'):
            with open(os.path.join(annotations, d), 'rt') as f:
                lines = [line.rstrip('\n') for line in f]
                line_id = 0
                while line_id < len(lines):
                    
                    # считывание изображение
                    image_path = lines[line_id]
                    image_path = image_path.replace('/', os.path.sep)
                 
                    line_id += 1
                    image_id = add_image(os.path.join(images, image_path) + '.jpg')
                    
                    # считывание аннотаций лиц
                    num_faces = int(lines[line_id])
                    line_id += 1
                    for i in range(num_faces):
                        params = [float(v) for v in lines[line_id].split()]

                        line_id += 1
                        start_x, start_y, end_x, end_y = ellipse_to_rect(params)
                        add_bbox(image_id, start_x, start_y, end_x, end_y)


# получение всех аннотаций лиц для заданного изображения
def get_faces(image_id):
    face_boxes = list(filter(lambda x : x['image_id'] == image_id, dataset['annotations']))
    return list(map(lambda x : x['bbox'], face_boxes))


# функция рассчета IOU для двух прямоугольных областей
def intersection_over_union(box_a, box_b):
    
	start_x = max(box_a[0], box_b[0])
	start_y = max(box_a[1], box_b[1])
	end_x = min(box_a[2], box_b[2])
	end_y = min(box_a[3], box_b[3])
    
    # площадь пересечения
	inter_area = max(0, end_x - start_x + 1) * max(0, end_y - start_y + 1)
	
	a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
	b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

	iou = inter_area / float(a_area + b_area - inter_area)

	return iou


# рассчет критериев для лиц одного изображения
# detector - передаваемый детектор
# image_id - id изображения из датасета
def compare_image(detector, image_id): 
    global all_count
    global iou_75
    global iou_50
    global iou_avg
    global false_25
    global non_det
    global total_time
    
    image = cv2.imread(dataset['images'][image_id]['file_name'])    
    
    start = datetime.now()
    faces_det = detector.calc_image(image, scale_factor=1.3, return_mode='box')
    end = datetime.now()
    total_time += (end - start).total_seconds()
    
    faces_true = get_faces(image_id)
    false_detections = 0
    
    for face_d in faces_det:
        
        ious = []
        for face_t in faces_true:
            
            iou = intersection_over_union(face_d, face_t)
            ious.append(iou)
        
        max_iou = max(ious)
            
        if max_iou >= 0.75:
            iou_75 += 1
                
        if max_iou >= 0.5:
            iou_50 += 1
                
        if max_iou <= 0.25:
            iou_avg += max_iou
            false_detections += 1
            false_25 += 1
            
        else:
            iou_avg += max_iou
            
    non_det += len(faces_true) - (len(faces_det) - false_detections)
    all_count += len(faces_true)


# рассчет критериев для изображений всего датасета
def compare_dataset():
    global all_count
    global iou_75
    global iou_50
    global iou_avg
    global false_25
    global non_det
    global total_time

    detector_lbp = DetectorLBP()
    
    s = ''
    for i in [2, 1, 0]:
        all_count = 0
        iou_75 = 0
        iou_50 = 0
        iou_avg = 0
        false_25 = 0
        non_det = 0
        total_time = 0
        
        count = 0
        for image_des in dataset['images']:
            count += 1
            compare_image(detector_lbp, image_des['id'])
            print(count, all_count / total_time)
            #print(i, count)
            
        s += str(i) + '\n'
        s += 'K1: ' + str(round(iou_75 / all_count, 3)) + '\n'
        s += 'K2:' + str(round(iou_50 / all_count, 3)) + '\n'
        s += 'K3:' + str(round(false_25 / all_count, 3)) + '\n'
        s += 'K4:' + str(round(non_det / all_count, 3)) + '\n'
        s += 'K5:' + str(round(iou_avg / all_count, 3)) + '\n'
        s += 'K6:' + str(round(iou_avg / (all_count - non_det), 3)) + '\n'
        s += 'FPS:' + str(round(len(dataset['images']) / total_time, 2)) + '\n\n'
        
    f = open('results.txt','w')    
    f.write(s)
    f.close()
    
fddb_dataset(args['ann'], args['pics'])
compare_dataset()

