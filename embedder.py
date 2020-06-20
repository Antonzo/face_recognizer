import pickle
import cv2
import os
from detectors import DetectorSSD, DetectorVJ, DetectorHOG, DetectorMMOD, DetectorLBP

# singleton-класс, инициализирующий извлекатель признаков лиц на основе CNN
class Embedder():
    
    # констурктор и инициализатор для создания singleton класса
    def __new__(cls, embedder_path='openface_nn4.small2.v1.t7', embeddings={}):
        
        if not hasattr(cls, 'instance'):
            cls.instance = super(Embedder, cls).__new__(cls)
        return cls.instance
    
    
    # embedder_path - путь к модели для извлечения признаков
    # embeddings - признаки в виде словаря, где ключи - имена
    def __init__(self, embedder_path='openface_nn4.small2.v1.t7', embeddings={}):
        
        if embedder_path != '':
            embedder_path = embedder_path.replace('/', os.path.sep)
            self.embedder = cv2.dnn.readNetFromTorch(embedder_path)
        else:
            self.embedder = None
        
        self.embeddings = embeddings
        
    
    # инициализация признаков напрямую
    def initialize_embeddings(self, embeddings={}):
        
        self.embeddings = embeddings
        
    
    # загрузка признаков из файла    
    def load_embeddings(self, embeddings_path='output/embeddings.pickle'):
        
        embeddings_path = embeddings_path.replace('/', os.path.sep)
        self.embeddings = pickle.loads(open(embeddings_path, 'rb').read())
        
    
    # загрузка признаковой модели из файла
    def load(self, embedder_path='openface_nn4.small2.v1.t7'):
        
        embedder_path = embedder_path.replace('/', os.path.sep)
        self.embedder = cv2.dnn.readNetFromTorch(embedder_path)
        
    
    def calc_face(self, face):
        
        # cоздание блоба лица (нормировка и сжатие) и затем 
        # пропускание блоба через признаковую модель
        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                        (96, 96), (0, 0, 0), swapRB=True, crop=False)
        
        self.embedder.setInput(faceBlob)
        vec = self.embedder.forward()
        
        return vec


    # Извлечение признаков из ROI лиц датасета
    # faces_roi содержит области лиц и соотв. им имена,
    # может быть типа DetectorCNN, словарь или строка
    # saving_path - путь для сохранения признаков, если '', 
    # то не сохраняются
    def calc_dataset(self, faces_roi='output/faces_roi_cnn.pickle', saving_path='output/embeddings.pickle'):
        
        faces_roi_t = type(faces_roi)
        
        if faces_roi_t in [DetectorSSD, DetectorVJ, DetectorHOG, DetectorMMOD, DetectorLBP]:
            faces_r = faces_roi.faces_roi
            
        elif faces_roi_t is dict:
            faces_r = faces_roi.faces_roi
            
        elif faces_roi_t is str:
            faces_roi = faces_roi.replace('/', os.path.sep)
            faces_r = pickle.loads(open(faces_roi, "rb").read())
            
        # списки для признаков и имен, взаимосоответствующие по индексу
        known_embeddings = []
        
        # для каждой ROI лица
        for index in range(len(faces_r['faces'])):
            
             # создание блоба для области интереса лица 
             # пропускание блоба через признаковую модель, 
             # чтобы получить 128-мерные признаки для лица
             face_blob = cv2.dnn.blobFromImage(faces_r['faces'][index], 1.0 / 255,
                        (96, 96), (0, 0, 0), swapRB=True, crop=False)
             self.embedder.setInput(face_blob)
             vec = self.embedder.forward()
             
             # разворачивание вектора в 1 измерение и добавление
             # в массив признаков
             known_embeddings.append(vec.flatten())
        
        # заполнение результирующего словаря
        self.embeddings = {'embeddings' : known_embeddings, 'names' : faces_r['names']}
        
        # сохранение признаков
        if saving_path != '':
            
            saving_path = saving_path.replace('/', os.path.sep)
            f = open(saving_path, 'wb')
            f.write(pickle.dumps(self.embeddings))
            f.close()
            
            
        