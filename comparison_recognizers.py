from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from imutils import paths
import cv2
import os
import image_effects as ie
import pickle
from embedder import Embedder
from detectors import DetectorSSD


# создание расширенного датасета из обычного и сохранение полученных признаков
def make_dataset(images_path='dataset'):
    
    
    detector = DetectorSSD()
    
    embedder = Embedder()
    
    images_path = images_path.replace('/', os.path.sep)
    
    ind_image_paths = list(paths.list_images(images_path))
    
    # списки для лиц и имен, взаимосоответствующие по индексу
    known_ems = []
    known_names = []
    
    for (i, image_path) in enumerate(ind_image_paths):
        print(i + 1)
        # получение имя из пути к файлу
        name = image_path.split(os.path.sep)[-2]

        image = cv2.imread(image_path)
     
        # ОБЫЧНОЕ
        temp_img = image.copy()
        face = detector.calc_image(temp_img)[0]

        embeddings = embedder.calc_face(face)
        known_ems.append(embeddings.flatten())
        known_names.append(name)
        
        # ТЕМНЕЕ 4
        for i in range(1, 5):
            temp_img = image.copy()
            face = detector.calc_image(ie.darken(temp_img, i/10))
            if len(face) > 0:
                embeddings = embedder.calc_face(face[0])
                known_ems.append(embeddings.flatten())
                known_names.append(name)

            
        # СВЕТЛЕЕ 4
        
        for i in range(1, 5):
            temp_img = image.copy()
            face = detector.calc_image(ie.brighten(temp_img, i/10))
            if len(face) > 0:
                embeddings = embedder.calc_face(face[0])
                known_ems.append(embeddings.flatten())
                known_names.append(name)

        # ДОЖДЬ 3
        temp_img = image.copy()
        face = detector.calc_image(ie.add_rain(temp_img, rain_type = 'drizzle'))
        if len(face) > 0:
            embeddings = embedder.calc_face(face[0])
            known_ems.append(embeddings.flatten())
            known_names.append(name)
            
        temp_img = image.copy()
        face = detector.calc_image(ie.add_rain(temp_img, rain_type = 'heavy'))
        if len(face) > 0:
            embeddings = embedder.calc_face(face[0])
            known_ems.append(embeddings.flatten())
            known_names.append(name)
            
        temp_img = image.copy()
        face = detector.calc_image(ie.add_rain(temp_img, rain_type = 'torrential'))
        if len(face) > 0:
            embeddings = embedder.calc_face(face[0])
            known_ems.append(embeddings.flatten())
            known_names.append(name)
            
        # ТУМАН 4
        for i in range(1, 5):
            temp_img = image.copy()
            face = detector.calc_image(ie.add_fog(temp_img, i/10))
            if len(face) > 0:
                embeddings = embedder.calc_face(face[0])
                known_ems.append(embeddings.flatten())
                known_names.append(name)
            
    embeddings = {'embeddings':known_ems, 'names': known_names}  
    f = open('weather_ems/ems_weather_4.pickle', 'wb')
    f.write(pickle.dumps(embeddings))
    f.close()      


    print(len(known_ems), len(known_names))
    return known_ems, known_names


# тренировка с перебором параметров SVC
def train_svc(X_train, y_train):
    
    res = ''
    grid1 = GridSearchCV(SVC(kernel='linear'), {'C': [i/10 for i in range(1, 200)]}, cv=25) 
    grid1.fit(X_train, y_train)
    res += 'SVC linear\n'
    res += str(grid1.best_score_) +  '\n'
    res += str(grid1.best_params_) + '\n\n'
    
    grid2 = GridSearchCV(SVC(kernel='poly'), {'C': [i/10 for i in range(1, 200)], 'degree' : [i for i in range(1, 5)],
                                              'gamma' : ['scale', 'auto'], 'coef0': [i/5 for i in range(1, 100)]}, cv=25) 
    grid2.fit(X_train, y_train)
    res += 'SVC poly\n'
    res += str(grid2.best_score_) +  '\n'
    res += str(grid2.best_params_) + '\n\n'
    
    grid3 = GridSearchCV(SVC(kernel='sigmoid'), {'C': [i/10 for i in range(1, 200)], 'gamma' : ['scale', 'auto'],
                                                 'coef0': [i/5 for i in range(1, 100)]}, cv=25) 
    grid3.fit(X_train, y_train)
    res += 'SVC sigmoid\n'
    res += str(grid3.best_score_) +  '\n'
    res += str(grid3.best_params_) + '\n\n'
    
    grid4 = GridSearchCV(SVC(kernel='rbf'), {'C': [i/100 + 0.001 for i in range(0, 1)], 'gamma' : ['scale', 'auto']}, cv=25) 
    grid4.fit(X_train, y_train)
    res += 'SVC rbf\n'
    res += str(grid3.best_score_) +  '\n'
    res += str(grid3.best_params_) + '\n\n'
    
    return res


# тренировка с перебором параметров SGD  
def train_sgd(X_train, y_train):

    res = ''
    grid1 = GridSearchCV(SGDClassifier(loss='hinge', penalty='elasticnet'), {'l1_ratio' : [i/100 for i in range(0, 101)]}, cv=25) 
    grid1.fit(X_train, names)
    res += 'SGD hinge\n'
    res += str(grid1.best_score_) +  '\n'
    res += str(grid1.best_params_) + '\n\n'
    
    grid2 = GridSearchCV(SGDClassifier(loss='log', penalty='elasticnet'), {'l1_ratio' : [i/100 for i in range(0, 101)]}, cv=25) 
    grid2.fit(X_train, names)
    res += 'SGD log\n'
    res += str(grid2.best_score_) +  '\n'
    res += str(grid2.best_params_) + '\n\n'
    
    grid3 = GridSearchCV(SGDClassifier(loss='modified_huber', penalty='elasticnet'), {'l1_ratio' : [i/100 for i in range(0, 101)]}, cv=25) 
    grid3.fit(X_train, names)
    res += 'SGD modified_huber\n'
    res += str(grid3.best_score_) +  '\n'
    res += str(grid3.best_params_) + '\n\n'

    grid4 = GridSearchCV(SGDClassifier(loss='squared_hinge', penalty='elasticnet'), {'l1_ratio' : [i/100 for i in range(0, 101)]}, cv=25) 
    grid4.fit(X_train, names)
    res += 'SGD squared_hinge\n'
    res += str(grid4.best_score_) +  '\n'
    res += str(grid4.best_params_) + '\n\n'
    
    grid5 = GridSearchCV(SGDClassifier(loss='perceptron', penalty='elasticnet'), {'l1_ratio' : [i/100 for i in range(0, 101)]}, cv=25) 
    grid5.fit(X_train, names)
    res += 'SGD perceptron\n'
    res += str(grid5.best_score_) +  '\n'
    res += str(grid5.best_params_) + '\n\n'
    
    return res
  
  
# тренировка с перебором параметров KNN  
def train_kn(X_train, y_train):    
    
    res = ''
    grid1 = GridSearchCV(KNeighborsClassifier(weights='uniform'), {'n_neighbors' : [i for i in range(1, 10)],
                                                                   'p' : [i for i in range(1, 10)]}, cv=25) 
    grid1.fit(X_train, names)
    res += 'KN uniform\n'
    res += str(grid1.best_score_) +  '\n'
    res += str(grid1.best_params_) + '\n\n'
    
    grid2 = GridSearchCV(KNeighborsClassifier(weights='distance'), {'n_neighbors' : [i for i in range(1, 10)],
                                                                    'p' : [i for i in range(1, 10)]}, cv=25) 
    grid2.fit(X_train, names)
    res += 'KN distance\n'
    res += str(grid2.best_score_) +  '\n'
    res += str(grid2.best_params_) + '\n\n'

    return res


# тренировка с перебором параметров NB  
def train_nb(X_train, y_train):
    
    res = ''
    grid1 = GridSearchCV(GaussianNB(), {}) 
    grid1.fit(X_train, y_train)
    res += 'Gaussian NB\n'
    res += str(grid1.best_score_) +  '\n'
    res += str(grid1.best_params_) + '\n\n'
    
    
    grid2 = GridSearchCV(BernoulliNB(), {'alpha' : [i/100 for i in range(1, 100)], 'binarize' : [i/100 for i in range(-200, 200)]}, cv=25) 
    grid2.fit(X_train, y_train)
    res += 'Bernoulli NB\n'
    res += str(grid2.best_score_) +  '\n'
    res += str(grid2.best_params_) + '\n\n'
    return res


# тренировка с перебором параметров методов, связанных с деревьями
def train_forest(X_train, y_train):
    
    res = ''
    grid1 = GridSearchCV(RandomForestClassifier(), {'n_estimators' : [i for i in range(1, 50)]}, cv=25)
    grid1.fit(X_train, y_train)
    res += 'Random Forest\n'
    res += str(grid1.best_score_) +  '\n'
    res += str(grid1.best_params_) + '\n\n'
    
    grid2 = GridSearchCV(ExtraTreesClassifier(), {'n_estimators' : [i for i in range(1, 50)]}, cv=25)
    grid2.fit(X_train, y_train)
    res += 'Extremely Randomized Trees\n'
    res += str(grid2.best_score_) +  '\n'
    res += str(grid2.best_params_) + '\n\n'
    
    grid3 = GridSearchCV(DecisionTreeClassifier(), {})
    grid3.fit(X_train, y_train)
    res += 'Random Forest\n'
    res += str(grid3.best_score_) +  '\n'
    res += str(grid3.best_params_) + '\n\n'
    return res

# признаки лиц из расширенного датасета для 2, 3, 4 классов
ems_paths = ['weather_ems/ems_weather_2.pickle', 
             'weather_ems/ems_weather_3.pickle',
             'weather_ems/ems_weather_4.pickle']

# подбор параметров для каждого из методов и запись в файл наилучших
classes = 2
for ems_path in ems_paths:
    
    embeddings = pickle.loads(open(ems_path, "rb").read())
        
    X_train = embeddings['embeddings']
    names = embeddings['names']
        
    le = LabelEncoder()
    y_train = le.fit_transform(names)
    res = ''
    
    res += train_svc(X_train, y_train)
    res += train_sgd(X_train, y_train)
    res += train_kn(X_train, y_train)
    res += train_nb(X_train, y_train)
    res += train_forest(X_train, y_train)
    
    f = open('res_weather_' + str(classes) + '.txt', 'w')
    f.write(res)
    f.close()
        
    classes += 1
    print(classes)
    
