import pickle
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from embedder import Embedder


# singleton-класс, инициализирующий распознаватель лиц на основе 
# метода опорных векторов
class Recognizer():
    
    # конструктор и инициализатор для создания singleton класса
    def __new__(cls, recognizer='output/recognizer.pickle', 
                 label_encoder='output/le.pickle'):
        
        if not hasattr(cls, 'instance'):
            cls.instance = super(Recognizer, cls).__new__(cls)
        return cls.instance
    
    
    # self.recognizer - распознаватель на основе метода опорных векторов
    # self.label_encoder - кодирует целевые классы со значениями от 0 до n_classes-1.
    def __init__(self, recognizer='output/recognizer.pickle', 
                 label_encoder='output/le.pickle'):
        
        if recognizer == None:
            #self.recognizer = SVC(C=1.0, kernel='linear', probability=True)
            self.recognizer = KNeighborsClassifier()
        elif type(recognizer) is str:
            recognizer = recognizer.replace('/', os.path.sep)
            self.recognizer = pickle.loads(open(recognizer, "rb").read())
        else: #TODO
            self.recognizer = recognizer
            
        if label_encoder == None:
            self.label_encoder = LabelEncoder()
        elif type(label_encoder) is str:
            label_encoder = label_encoder.replace('/', os.path.sep)
            self.label_encoder = pickle.loads(open(label_encoder, "rb").read())
        else:
            self.label_encoder = label_encoder
        
    
    # загрузка распознавателя из файла
    def load_recognizer(self, recognizer_path='output/recognizer.pickle'):
        
        recognizer_path = recognizer_path.replace('/', os.path.sep)
        self.recognizer = pickle.loads(open(recognizer_path, "rb").read())
        
    
    # загрузка кодировщика классов из файла
    def load_label_encoder(self, label_encoder_path='output/le.pickle'):
        
        label_encoder_path = label_encoder_path.replace('/', os.path.sep)
        self.label_encoder = pickle.loads(open(label_encoder_path, 'rb').read())
        
    
    # загрузка распознавателя и кодировщика классов из файлов
    def load(self, recognizer_path='output/recognizer.pickle',
                    label_encoder_path='output/le.pickle'):
        
        self.load_recognizer(recognizer_path)
        self.load_label_encoder(label_encoder_path)
    
    
    def recognize(self, embeddings):
        
         # выполнение классификации
         preds = self.recognizer.predict_proba(embeddings)[0]
            
         # получение наиболее подходящего класса 
         j = np.argmax(preds)
                
         # получение имени этого класса
         name = self.label_encoder.classes_[j]
         
         return name
            
            
    # тренировка распознавателя
    # embeddings - признаки или признаковая модель их содержащая
    # saving_path_rec - сохранение обученного распознавателя
    # saving_path_le - сохранение кодировщика классов
    def train(self, embeddings='output/embeddings.pickle', saving_path_rec='output/recognizer.pickle', 
                                            saving_path_le='output/le.pickle'):
        
        saving_path_rec = saving_path_rec.replace('/', os.path.sep)
        saving_path_le = saving_path_le.replace('/', os.path.sep)
        
        embeddings_t = type(embeddings)
        
        if embeddings_t is Embedder:
            embed = embeddings.embeddings
            
        elif embeddings_t is dict:
            embed = embeddings
        
        elif embeddings_t is str:
            embed = pickle.loads(open(embeddings, 'rb').read())
            
        # кодирование классов
        labels = self.label_encoder.fit_transform(embed['names'])

        # обучение распознавателя на 128-мерных признаках
        self.recognizer.fit(embed['embeddings'], labels)
        
        # сохранение рапознавателя
        if saving_path_rec != '':
                
            f = open(saving_path_rec, 'wb')
            f.write(pickle.dumps(self.recognizer))
            f.close()

        # сохранение кодировщика меток
        if saving_path_le != '': 
                
            f = open(saving_path_le, "wb")
            f.write(pickle.dumps(self.label_encoder))
            f.close()
    
    
    