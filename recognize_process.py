from video_stream import VideoStream
from fps import FPS
import numpy as np
import time
import cv2
import os
from detectors import DetectorSSD, DetectorVJ, DetectorHOG, DetectorMMOD, DetectorLBP
from embedder import Embedder
from recognizer import Recognizer


# проверка на тип детектора
def is_detector(detector):
    
    d_types = [DetectorSSD, DetectorVJ, DetectorHOG, DetectorMMOD, DetectorLBP]
    if type(detector) in d_types:
        return True
    return False


# функция распознавания лиц на изображении 
# detector - детектор одного из типов
# embedder - извлекатель признаков
# recognizer - классификатор признаков
# image - изображение, на котором необходимо распознать лица
# detector_params - список параметров для детектора, если 
# 'default', то используются параметры по умолчанию
# showing - показывать ли изображение
# saving_path - путь для сохранения изображения, если пустая строка, 
# то не сохраняется
def recognize_image(detector, embedder:Embedder, recognizer:Recognizer,
                    image, detector_params = 'default', showing=True, saving_path=''):
    
    image_t = type(image)
    
    if image_t is str:
        image = image.replace('/', os.path.sep)
        image = cv2.imread(image)
        
    elif image_t is list:
        image = np.array(image)
        
    if not is_detector(detector):
        raise TypeError('Incorrect type of detector')
     
    # вызов детектора с заданными параметрами
    if detector_params == 'default':
        faces_roi, boxes = detector.calc_image(image, return_mode='both')
        
    elif type(detector) == DetectorSSD:
        confidence = detector_params[0]
        faces_roi, boxes = detector.calc_image(image, confidence=confidence, return_mode='both')
        
    elif type(detector) == DetectorVJ or type(detector) == DetectorLBP:
        [scale_factor, min_neighbors] = detector_params
        faces_roi, boxes = detector.calc_image(image, scale_factor=scale_factor,
                                               min_neighbors=min_neighbors, return_mode='both')
        
    elif type(detector) == DetectorHOG or type(detector) == DetectorMMOD:
        upsampling_times = detector_params[0]
        faces_roi, boxes = detector.calc_image(image, upsampling_times=upsampling_times, 
                                               return_mode='both')
        

    # для каждого обнаруженного лица
    for i in range(len(faces_roi)):
        
        # получение признаков
        embeddings = embedder.calc_face(faces_roi[i])    
        
        # получение класса
        name = recognizer.recognize(embeddings)  
        start_x, start_y, end_x, end_y = boxes[i]
        
        # рисование ограничительной рамки лица, написание имени
        text = '{}'.format(name)
        y = start_y - 10 if start_y - 10 > 10 else start_y + 10
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y),
                         (255, 0, 0), 4)
        cv2.putText(image, text, (start_x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 4)
        
    if saving_path != '':
        
        cv2.imwrite(saving_path, image)
        
    # если нужно вывести изоражение на экран
    if showing:

        cv2.imshow('Image', image)
        cv2.waitKey(0)
        


# функция распознавания лиц на видеопотоке
# detector - детектор одного из типов
# embedder - извлекатель признаков
# recognizer - классификатор признаков
# detector_params - список параметров для детектора, если 
# 'default', то используются параметры по умолчанию
# source - идентификатор устройства
def recognize_video(detector, embedder:Embedder, recognizer:Recognizer, detector_params = 'default', source=0):
    
    # инициализация видеопотока
    print('Starting video stream...')
    vs = VideoStream(src=source).start()
    
    if not is_detector(detector):
        raise TypeError('Incorrect type of detector')
        
    # разогрев камеры
    time.sleep(0.5)

        
    # запуск оценщика пропускной способности FPS
    fps = FPS().start()

    # цикл по фреймам из видео 
    while True:
        
        frame = vs.read()
        
        if detector_params == 'default':
            faces_roi, boxes = detector.calc_image(frame, return_mode='both')
        
        elif type(detector) == DetectorSSD:
            confidence = detector_params[0]
            faces_roi, boxes = detector.calc_image(frame, confidence=confidence, return_mode='both')
        
        elif type(detector) == DetectorVJ or type(detector) == DetectorLBP:
            [scale_factor, min_neighbors] = detector_params
            faces_roi, boxes = detector.calc_image(frame, scale_factor=scale_factor,
                                               min_neighbors=min_neighbors, return_mode='both')
        
        elif type(detector) == DetectorHOG or type(detector) == DetectorMMOD:
            upsampling_times = detector_params[0]
            faces_roi, boxes = detector.calc_image(frame, upsampling_times=upsampling_times, 
                                               return_mode='both')

        for i in range(len(faces_roi)):
            
            embeddings = embedder.calc_face(faces_roi[i])    
            name = recognizer.recognize(embeddings)  
            start_x, start_y, end_x, end_y = boxes[i]
        
            text = '{}'.format(name)
            y = start_y - 10 if start_y - 10 > 10 else start_y + 10
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y),
                             (0, 0, 255), 2)
            cv2.putText(frame, text, (start_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # обновление счетчика FPS
        fps.update()
        
        # показ выходного фрейма
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF
        
        # завершение при нажатии 'q'
        if key == ord("q"):
            break

    fps.stop()
    print('Elasped time: {:.2f}'.format(fps.elapsed()))
    print('Approx. FPS: {:.2f}'.format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()