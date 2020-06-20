from imutils import paths
import numpy as np
import pickle
import cv2
import dlib
import os


# singleton-класс, инициализирующий детектор лиц на основе CNN
class DetectorSSD():
    
    # констурктор и инициализатор для создания singleton класса
    def __new__(cls, proto_path='face_detection_cnn/deploy.prototxt',
                model_path='face_detection_cnn/weights.caffemodel',
                faces_roi={}):
        
        if not hasattr(cls, 'instance'):
            cls.instance = super(DetectorSSD, cls).__new__(cls)
        return cls.instance
    
    
    # self.detector - стандартный класс Net из opencv для нейронных сетей
    # self.faces_roi - ROI лиц, определенных детектором (как правило, результат
    # работы детектора) в формате словаря, где ключ - имя, а значение - ROI лица
    # proto_path - архитектура модели
    # model_path - веса модели
    def __init__(self, proto_path='face_detection_cnn/deploy.prototxt', 
                 model_path='face_detection_cnn/weights.caffemodel',
                 faces_roi={}):
        
        if proto_path != '' and model_path != '':
            
            proto_path = proto_path.replace('/', os.path.sep)
            model_path = model_path.replace('/', os.path.sep)
            self.detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            
        else:
            self.detector = None
        
        self.faces_roi = faces_roi
        
        
    # инициализация ROI лиц напрямую
    def initialize_fr(self, faces_roi={}):
        
        self.faces_roi = faces_roi
        
        
    # загрузка ROI лиц из указанного файла
    def load_fr(self, faces_roi_path='output/faces_roi_cnn.pickle'):
        
        faces_roi_path = faces_roi_path.replace('/', os.path.sep)
        self.faces_roi = pickle.loads(open(faces_roi_path, 'rb').read())
        
        
    # загрузка детектора лиц из файлов
    def load(self, proto_path='face_detection_cnn/deploy.prototxt', 
                 model_path='face_detection_cnn/weights.caffemodel'):

        proto_path = proto_path.replace('/', os.path.sep)
        model_path = model_path.replace('/', os.path.sep)
        self.detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        

    def calc_image(self, image, return_mode = 'image', confidence=0.5):
        
    
        blue, green, red = [104.0, 177.0, 123.0] #[123.0, 177.0, 104.0]
    
        (h, w) = image.shape[:2]
             
        # конструирование блоба, то есть вычитание средних значений
        image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (red, green, blue), swapRB=False, crop=False) 
        
        # применение детектора лиц OpenCV на основе глубокого обучения, 
        # чтобы локализовать лица во входном изображении
        self.detector.setInput(image_blob)
        detections = self.detector.forward()
        
        all_faces = []
        all_boxes = []
        
        # цикл по всем обнаруженным лицам
        for i in range(0, detections.shape[2]):
        
            # извлечение вероятности того, что обнаружено лицо
            conf = detections[0, 0, i, 2]
            
            if conf > confidence:
            
                # получение верхней левой и нижней правой точек ROI лица
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")
                
                # извлечение ROI лица
                face = image[start_y:end_y, start_x:end_x]
                (f_h, f_w) = face.shape[:2]
                
                # удостоверение в том, что лицо достаточного размера
                if f_w > 20 and f_h > 20:
                    all_faces.append(face)
                    all_boxes.append((start_x, start_y, end_x, end_y))
                    
        if return_mode == 'image':       
            return all_faces
        
        elif return_mode == 'box':
            return all_boxes
        
        elif return_mode == 'both':
            return all_faces, all_boxes
        
        
    # вычисление ROI лиц для датасета
    # dataset_path - путь к датасету, состоящему из папок персон, содержащих изображения
    # saving_path - путь для сохранения полученных результатов, если '', то результаты не сохр.
    # confidence - уверенность (вероятность того, что это лицо) при распознавании
    def calc_dataset(self, dataset_path='dataset', saving_path='output/faces_roi_cnn.pickle',
                     confidence=0.5):
        
        dataset_path = dataset_path.replace('/', os.path.sep)
        image_paths = list(paths.list_images(dataset_path))
        
        # списки для лиц и имен, взаимосоответствующие по индексу
        known_faces = []
        known_names = []
        
        blue, green, red = [104.0, 177.0, 123.0] #[123.0, 177.0, 104.0]
        
        # цикл по всем папкам в датасете
        for (i, image_path) in enumerate(image_paths):
            
            # получение имя из пути к файлу
            name = image_path.split(os.path.sep)[-2]
            image = cv2.imread(image_path)
            
            #image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]
             
            # конструирование блоба, то есть вычитание средних значений
            image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (red, green, blue), swapRB=False, crop=False)
             
            # применение детектора лиц OpenCV на основе глубокого обучения, 
            # чтобы локализовать лица во входном изображении
            self.detector.setInput(image_blob)
            detections = self.detector.forward()
             
            # убеждение в том, что хотя бы 1 лицо найдено
            if len(detections) > 0:
                
                # предполагается, что изображение имеет только 1 лицо
                i = np.argmax(detections[0, 0, :, 2])
                conf = detections[0, 0, i, 2]
                  
                # проверка достаточности для вероятности обнаружения лица
                if conf > confidence:
                    
                    # вычисление координат ограничительной рамки для лица
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (start_x, start_y, end_x, end_y) = box.astype('int')
                    
                    # ивлечение ROI лица
                    face = image[start_y:end_y, start_x:end_x]
                    (f_h, f_w) = face.shape[:2]
                    
                    # убеждение в том, что размеры лица достаточно велики
                    if f_w > 20 and f_h > 20:
                        known_names.append(name)
                        known_faces.append(face)
                        
        # заполнение результирующего словаря
        self.faces_roi = {'faces': known_faces, 'names': known_names}
        
        
        # если требуется сохранить полученные ROI лиц
        if saving_path != '':

            saving_path = saving_path.replace('/', os.path.sep)
            f = open(saving_path, 'wb')
            f.write(pickle.dumps(self.faces_roi))
            f.close()
            

# singleton-класс, инициализирующий детектор лиц на основе VJ
class DetectorVJ():
    
    # конструктор и инициализатор для создания singleton класса
    # model_path - путь к модели в формате .xml
    # faces_roi - словарь, содержащий ROI лиц и соотв. им имена персон
    def __new__(cls, model_path='haarcascade_frontalface_default.xml',
                faces_roi={}):
        
        if not hasattr(cls, 'instance'):
            cls.instance = super(DetectorVJ, cls).__new__(cls)
        return cls.instance
    
    # self.detector - непосредственно детектор 
    def __init__(self, model_path='haarcascade_frontalface_default.xml', 
                 faces_roi={}):
        
        if model_path != None:
            model_path = model_path.replace('/', os.path.sep)
            self.detector = cv2.CascadeClassifier(model_path)
            
        else:
            self.detector = None
            
        self.faces_roi = faces_roi 
        
        
    # инициализация ROI лиц напрямую
    def initialize_fr(self, faces_roi={}):

        self.faces_roi = faces_roi
        
        
    # загрузка ROI лиц из указанного файла
    def load_fr(self, faces_roi_path='output/faces_roi_haar.pickle'):
        
        faces_roi_path = faces_roi_path.replace('/', os.path.sep)
        self.faces_roi = pickle.loads(open(faces_roi_path, 'rb').read())
    
    
    def calc_image(self, image, scale_factor=1.3,
                   min_neighbors=5, return_mode = 'image'):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detectMultiScale(gray, scale_factor, min_neighbors)
        
        all_faces = []
        all_boxes = []
        
        for (x, y, f_w, f_h) in detections:
            
            face = image[y: y + f_h, x: x + f_w]
            (start_x, start_y, end_x, end_y) = x, y, x + f_w, y + f_h
            
            all_faces.append(face)
            all_boxes.append((start_x, start_y, end_x, end_y))
            
        if return_mode == 'image':       
            return all_faces
        
        elif return_mode == 'box':
            return all_boxes
        
        elif return_mode == 'both':
            return all_faces, all_boxes
        
        
        
    
    # вычисление ROI лиц для датасета
    def calc_dataset(self, dataset_path='dataset', saving_path='output/faces_roi_haar.pickle',
                     scale_factor=1.3, min_neighbors=5):
        
        dataset_path = dataset_path.replace('/', os.path.sep)
        image_paths = list(paths.list_images(dataset_path))
        
        # списки для лиц и имен, взаимосоответствующие по индексу
        known_faces = []
        known_names = []
        
        # цикл по всем папкам в датасете
        for (i, image_path) in enumerate(image_paths):
            
            # получение имя из пути к файлу
            name = image_path.split(os.path.sep)[-2]
            image = cv2.imread(image_path)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.detector.detectMultiScale(gray, scale_factor, min_neighbors)
            
            # убеждение в том, что хотя бы 1 лицо найдено
            if len(detections) > 0:
                
                # предполагается, что изображение имеет только 1 лицо
                (x, y, f_w, f_h) = detections[0]
                
                # убеждение в том, что размеры лица достаточно велики
                if f_w > 20 and f_h > 20:
                    face = image[y: y + f_h, x: x + f_w]
                    known_names.append(name)
                    known_faces.append(face)
                    
        # заполнение результирующего словаря
        self.faces_roi = {'faces': known_faces, 'names': known_names}
        
        
        # если требуется сохранить полученные ROI лиц
        if saving_path != '':
            
            saving_path = saving_path.replace('/', os.path.sep)
            f = open(saving_path, 'wb')
            f.write(pickle.dumps(self.faces_roi))
            f.close()
                
                        

# singleton-класс, инициализирующий детектор лиц на основе HOG
class DetectorHOG():

    # констурктор и инициализатор для создания singleton класса
    def __new__(cls, faces_roi={}):
        
        if not hasattr(cls, 'instance'):
            cls.instance = super(DetectorHOG, cls).__new__(cls)
        return cls.instance
    
    
    def __init__(self, faces_roi={}):
        
       self.detector = dlib.get_frontal_face_detector()
       self.faces_roi = faces_roi 
       
       
    # инициализация ROI лиц напрямую
    def initialize_fr(self, faces_roi={}):

        self.faces_roi = faces_roi
     
        
    # загрузка ROI лиц из указанного файла
    def load_fr(self, faces_roi_path='output/faces_roi_hog.pickle'):
        
        faces_roi_path = faces_roi_path.replace('/', os.path.sep)
        self.faces_roi = pickle.loads(open(faces_roi_path, 'rb').read())
      
        
    def calc_image(self, image, upsampling_times=0, return_mode = 'image'):
        
        all_faces = []
        all_boxes = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

        detections = self.detector(gray, upsampling_times)


        for detection in detections:
            
            start_x = detection.left()
            start_y = detection.top()
            end_x = detection.right()
            end_y = detection.bottom()
            
            face = image[start_y: end_y, start_x: end_x]
            all_faces.append(face)
            all_boxes.append((start_x, start_y, end_x, end_y))  
                
        if return_mode == 'image':       
            return all_faces
        
        elif return_mode == 'box':
            return all_boxes
        
        elif return_mode == 'both':
            return all_faces, all_boxes
    
    
    # вычисление ROI лиц для датасета
    def calc_dataset(self, dataset_path='dataset', saving_path='output/faces_roi_hog.pickle',
                    upsampling_times=0):
        
        dataset_path = dataset_path.replace('/', os.path.sep)
        image_paths = list(paths.list_images(dataset_path))
        
        # списки для лиц и имен, взаимосоответствующие по индексу
        known_faces = []
        known_names = []
        
        # цикл по всем папкам в датасете
        for (i, image_path) in enumerate(image_paths):
            
            # получение имя из пути к файлу
            name = image_path.split(os.path.sep)[-2]
            image = cv2.imread(image_path)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.detector(gray, upsampling_times)
            
            # убеждение в том, что хотя бы 1 лицо найдено
            if len(detections) > 0:
                
                # предполагается, что изображение имеет только 1 лицо
                detection = detections[0]
                
                start_x = detection.left()
                start_y = detection.top()
                end_x = detection.right()
                end_y = detection.bottom()
                    
                # убеждение в том, что размеры лица достаточно велики
                if end_x - start_x > 20 and end_y - start_y > 20:
                    
                    face = image[start_y: end_y, start_x: end_x]
                    known_names.append(name)
                    known_faces.append(face)
                    
        # заполнение результирующего словаря
        self.faces_roi = {'faces': known_faces, 'names': known_names}
        
        
        # если требуется сохранить полученные ROI лиц
        if saving_path != '':
            
            saving_path = saving_path.replace('/', os.path.sep)
            f = open(saving_path, 'wb')
            f.write(pickle.dumps(self.faces_roi))
            f.close()
    

# singleton-класс, инициализирующий детектор лиц на основе MMOD
class DetectorMMOD():
    
    # констурктор и инициализатор для создания singleton класса
    def __new__(cls, model_path='mmod_human_face_detector.dat', faces_roi={}):
        
        if not hasattr(cls, 'instance'):
            cls.instance = super(DetectorMMOD, cls).__new__(cls)
        return cls.instance
    
    
    def __init__(self, model_path='mmod_human_face_detector.dat', faces_roi={}):

       self.detector = dlib.cnn_face_detection_model_v1(model_path)
       self.faces_roi = faces_roi 
    
    
    # инициализация ROI лиц напрямую
    def initialize_fr(self, faces_roi={}):

        self.faces_roi = faces_roi
     
        
    # загрузка ROI лиц из указанного файла
    def load_fr(self, faces_roi_path='output/faces_roi_mmod.pickle'):
        
        faces_roi_path = faces_roi_path.replace('/', os.path.sep)
        self.faces_roi = pickle.loads(open(faces_roi_path, 'rb').read())
        
        
    def calc_image(self, image, upsampling_times=0, return_mode = 'image'):
        
        all_faces = []
        all_boxes = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

        detections = self.detector(gray, upsampling_times)


        for detection in detections:
            start_x = detection.rect.left()
            start_y = detection.rect.top()
            end_x = detection.rect.right()
            end_y = detection.rect.bottom()
            
            face = image[start_y: end_y, start_x: end_x]
            all_faces.append(face)
            all_boxes.append((start_x, start_y, end_x, end_y))  
                
        if return_mode == 'image':       
            return all_faces
        
        elif return_mode == 'box':
            return all_boxes
        
        elif return_mode == 'both':
            return all_faces, all_boxes
     
        
    # вычисление ROI лиц для датасета
    def calc_dataset(self, dataset_path='dataset', saving_path='output/faces_roi_mmod.pickle',
                    upsampling_times=0):
        
        dataset_path = dataset_path.replace('/', os.path.sep)
        image_paths = list(paths.list_images(dataset_path))
        
        # списки для лиц и имен, взаимосоответствующие по индексу
        known_faces = []
        known_names = []
        
        # цикл по всем папкам в датасете
        for (i, image_path) in enumerate(image_paths):
            
            # получение имя из пути к файлу
            name = image_path.split(os.path.sep)[-2]
            image = cv2.imread(image_path)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.detector(gray, upsampling_times)
            
            # убеждение в том, что хотя бы 1 лицо найдено
            if len(detections) > 0:
                
                # предполагается, что изображение имеет только 1 лицо
                detection = detections[0]
                
                start_x = detection.rect.left()
                start_y = detection.rect.top()
                end_x = detection.rect.right()
                end_y = detection.rect.bottom()
                    
                # убеждение в том, что размеры лица достаточно велики
                if end_x - start_x > 20 and end_y - start_y > 20:
                    
                    face = image[start_y: end_y, start_x: end_x]
                    known_names.append(name)
                    known_faces.append(face)
                    
        # заполнение результирующего словаря
        self.faces_roi = {'faces': known_faces, 'names': known_names}
        
        
        # если требуется сохранить полученные ROI лиц
        if saving_path != '':
            
            saving_path = saving_path.replace('/', os.path.sep)
            f = open(saving_path, 'wb')
            f.write(pickle.dumps(self.faces_roi))
            f.close()
    

# singleton-класс, инициализирующий детектор лиц на основе LBP    
class DetectorLBP():
    
    # констурктор и инициализатор для создания singleton класса
    def __new__(cls, model_path='lbpcascade_frontalface.xml',
                faces_roi={}):
        
        if not hasattr(cls, 'instance'):
            cls.instance = super(DetectorLBP, cls).__new__(cls)
        return cls.instance
    
    def __init__(self, model_path='lbpcascade_frontalface.xml', 
                 faces_roi={}):
        
        if model_path != None:
            model_path = model_path.replace('/', os.path.sep)
            self.detector = cv2.CascadeClassifier(model_path)
            
        else:
            self.detector = None
            
        self.faces_roi = faces_roi 
        
        
    # инициализация ROI лиц напрямую
    def initialize_fr(self, faces_roi={}):

        self.faces_roi = faces_roi
        
        
    # загрузка ROI лиц из указанного файла
    def load_fr(self, faces_roi_path='output/faces_roi_lbp.pickle'):
        
        faces_roi_path = faces_roi_path.replace('/', os.path.sep)
        self.faces_roi = pickle.loads(open(faces_roi_path, 'rb').read())
    
    
    def calc_image(self, image, scale_factor=1.15,
                   min_neighbors=5, return_mode = 'image'):
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detectMultiScale(gray, scale_factor, min_neighbors)
        
        all_faces = []
        all_boxes = []
        
        for (x, y, f_w, f_h) in detections:
            
            face = image[y: y + f_h, x: x + f_w]
            (start_x, start_y, end_x, end_y) = x, y, x + f_w, y + f_h
            
            all_faces.append(face)
            all_boxes.append((start_x, start_y, end_x, end_y))
            
        if return_mode == 'image':       
            return all_faces
        
        elif return_mode == 'box':
            return all_boxes
        
        elif return_mode == 'both':
            return all_faces, all_boxes
        
        
        
    
    # вычисление ROI лиц для датасета
    def calc_dataset(self, dataset_path='dataset', saving_path='output/faces_roi_lbp.pickle',
                     scale_factor=1.3, min_neighbors=5):
        
        dataset_path = dataset_path.replace('/', os.path.sep)
        image_paths = list(paths.list_images(dataset_path))
        
        # списки для лиц и имен, взаимосоответствующие по индексу
        known_faces = []
        known_names = []
        
        # цикл по всем папкам в датасете
        for (i, image_path) in enumerate(image_paths):
            
            # получение имя из пути к файлу
            name = image_path.split(os.path.sep)[-2]
            image = cv2.imread(image_path)
            #image = imutils.resize(image, width=600)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.detector.detectMultiScale(gray, scale_factor, min_neighbors)
            
            # убеждение в том, что хотя бы 1 лицо найдено
            if len(detections) > 0:
                
                # предполагается, что изображение имеет только 1 лицо
                (x, y, f_w, f_h) = detections[0]
                
                # убеждение в том, что размеры лица достаточно велики
                if f_w > 20 and f_h > 20:
                    face = image[y: y + f_h, x: x + f_w]
                    known_names.append(name)
                    known_faces.append(face)
                    
        # заполнение результирующего словаря
        self.faces_roi = {'faces': known_faces, 'names': known_names}
        
        
        # если требуется сохранить полученные ROI лиц
        if saving_path != '':
            
            saving_path = saving_path.replace('/', os.path.sep)
            f = open(saving_path, 'wb')
            f.write(pickle.dumps(self.faces_roi))
            f.close()    