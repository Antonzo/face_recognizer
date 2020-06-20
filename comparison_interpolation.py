import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
import time

# список путей 
image_paths = ['images/anton_zamay.jpg', 'images/group_1.jpg', 
               'images/nikita_severin.jpg', 'images/yana_demyanenko.jpg']

# список алгоритмов интерполяции
interpolation_algorithms = [ 
    ('ближ. сосед', cv2.INTER_NEAREST), 
    ('билинейная', cv2.INTER_LINEAR), 
    ('бикубическая', cv2.INTER_CUBIC), 
    ('площадь пикселя', cv2.INTER_AREA),
    ('Ланцош', cv2.INTER_LANCZOS4)
]

# факторы масштабирования
factors = [i/10 for i in range(1, 10)]

# интерполяция изображения
# img - изображение для интерполяции
# factor - масштаб интерполяции (>0)
# is_plot - рисовать ли график
# file_name - сохранение графика, если '', то не сохранять
def inter_image(img, factor, is_plot=True, file_name='', info=False):
    
    # подсчет размеров изображений
    height, width, channels = img.shape
    height_res, width_res = int(height * factor), int(width * factor)
    
    # печать информации о изображении
    if info:
        print('orig size:', height, width)
        print('scale factor:', factor)
        print('resize size:', height_res, width_res)

    # применение к изображению каждого алгоритма интерполяции
    imgs_res = []
    for alg in interpolation_algorithms:
        img_res = cv2.resize(img, (width_res, height_res), interpolation = alg[1])
        imgs_res.append(img_res)
    
    # рисование графика
    if is_plot:
        
        # ширина и высота всего рисунка в дюймах
        plt.figure(figsize=(11, 2))
        
        # добавление подграфика к текущей фигуре
        # кол-во строк, кол-во столбцов, индекс текущего подграфика
        plt.subplot(1, len(imgs_res) + 1, 1)
        plt.title('оригинал')
        plt.imshow(img)
        
        # выключение линий осей
        plt.axis('off')
        
        # для каждой интерполяции добавляем ее в общую фигуру
        # вместе с названием алгоритма
        for i in range(len(imgs_res)):
            plt.subplot(1, len(imgs_res) + 1, i + 2)
            plt.title(interpolation_algorithms[i][0])
            plt.imshow(imgs_res[i])
            plt.axis('off')
         
        # иширина между подграфиками
        plt.subplots_adjust(wspace = 0)
        
        # сохранние фигуры в файл
        if file_name != '':
            plt.savefig(file_name)
            
        plt.show()
    
    return imgs_res

# интерполяция множества картинок по множеству определенных факторов 
# image_paths - список путей к изображениям
# factors - список факторов (>0 каждый)
# count - количество применений каждой интерполяции к каждому изображению
# mode - режим, либо для среднего времени либо для суммарного
def inter_dataset(image_paths, factors, count=5, mode='mean'):
    
    # список загруженных изображений
    images = []

    for i in range(len(image_paths)):
    
        # для поддержки с любой ОС
        image_paths[i] = image_paths[i].replace('/', os.path.sep)
        images.append(cv2.cvtColor(cv2.imread(image_paths[i]),  cv2.COLOR_BGR2RGB))
                
    data = []
    for alg in interpolation_algorithms:
        for factor in factors:
            times = []
            
            for image in images:   
                for i in range(count):
                    t_0 = time.time()
                    cv2.resize(images[0], None, fx = factor, fy = factor, interpolation = alg[1])
                    t_delta = (time.time() - t_0)
                    times.append(t_delta)

            mean_time = np.mean(times)
            sum_time = np.sum(times)
            dispersion = 2 * np.std(times)
            
            if mode == 'mean':
                mode_time = mean_time
            elif mode == 'sum':
                mode_time = sum_time
                
            # словарь со значениями времени, алгоритма, отклонения, фактора

            d = dict(time = mode_time, algorithm = alg[0], dispersion = dispersion, scale = factor)
  
            data.append(d)
                
    df = pd.DataFrame(data)
    ax = df.set_index('scale').groupby('algorithm')['time'].plot(legend=True, figsize=(10, 8), grid=True, title='Сравнение времени работы алгоритмов')
    ax[0].set_xlabel('фактор масштабирования')
    
    print(type(ax))
    if mode == 'mean':
        ax[0].set_ylabel('среднее время (сек)')
    elif mode == 'sum':
        ax[0].set_ylabel('общее время (сек) для {} изображений'.format(len(images) * len(factors) * count))
        

img = cv2.imread("images/elon_mask.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
inter_image(img, 0.7, is_plot=True, file_name='', info=False)

