from detectors import DetectorSSD, DetectorVJ, DetectorHOG, DetectorMMOD, DetectorLBP
from embedder import Embedder
from recognizer import Recognizer
import recognize_process as rp



detector_vj = DetectorVJ()
detector_ssd = DetectorSSD()
detector_hog = DetectorHOG()
detector_mmod = DetectorMMOD()
detector_lbp = DetectorLBP()
embedder = Embedder()
recognizer = Recognizer()

    
# вычисление областей ROI всеми детекторами на датасете
def calc_all_detectors():
    
    print('Haar detector calculating...')
    detector_vj.calc_dataset()
    
    print('SSD detector calculating...')
    detector_ssd.calc_dataset()
    
    print('HOG detector calculating...')
    detector_hog.calc_dataset()
    
    print('MMOD detector calculating...')
    detector_mmod.calc_dataset()
    
    print('LBP detector calculating...')
    detector_lbp.calc_dataset()
    

# извлечение признаков
# faces_roi - ROI какого детектора использовать
def calc_embeddings(faces_roi='output/faces_roi_cnn.pickle'):
    
    print('Embedder calculating...')
    embedder.calc_dataset(faces_roi=faces_roi)
    
    
# загрузка уже вычисленных ROI лиц 
def load_all_fr():
    
    print('Loading faces ROI...')
    detector_vj.load_fr()
    detector_ssd.load_fr()
    detector_hog.load_fr()
    detector_mmod.load_fr()
    detector_lbp.load_fr()
    

# загрузка признаков
def load_ems():

    print('Loading embeddings...')
    embedder.load_embeddings()
    

# тренировка распознавателя с признаками по умолчанию (из сериализованного файла)
def train_rec():
    
    print('Training recognizer...')
    recognizer.train()
    

# загрузка распознавателя и кодировщика классов
def load_rec():
    
    print('Training recognizer...')
    recognizer.load()


# пример работы программы 
# train - тренировать ли модели
# var - если False, то для фото
# если True, то для видео
def example(train=False, var=False):
    
    if train:
        detector_ssd.calc_dataset()
        embedder.calc_dataset()
        recognizer.train()
        
    else:
        detector_ssd.load()
        embedder.load_embeddings()
        recognizer.load()
    
    if var:
        rp.recognize_video(detector_ssd, embedder, recognizer)
        
    else:
        rp.recognize_image(detector_ssd, embedder, recognizer, 'images/nastya_gromenko.jpg')
    
    

example(var=True)
    