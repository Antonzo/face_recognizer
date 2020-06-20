import cv2 as cv2
import numpy as np
import random


# меняет яркость изображения на указанный коэффициент
def change_light(image, light_coeff):
    
    image_HLS = cv2.cvtColor(image,cv2.COLOR_BGR2HLS) 
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    image_HLS[:,:,1] = image_HLS[:,:,1] * light_coeff
    
    if light_coeff > 1:
        image_HLS[:,:,1][image_HLS[:,:,1] > 255]  = 255 
    else:
        image_HLS[:,:,1][image_HLS[:,:,1] < 0] = 0
        
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_BGR = cv2.cvtColor(image_HLS ,cv2.COLOR_HLS2BGR) 
    
    return image_BGR


# генерирует координаты для всех накладываемых теней
def generate_shadow_coordinates(no_of_shadows, rectangular_roi, shadow_dimension):
    
    vertices_list = []
    (x1, y1, x2, y2) = rectangular_roi
    
    for index in range(no_of_shadows):
        vertex=[]
        
        for dimensions in range(shadow_dimension): 
            vertex.append((random.randint(x1, x2), random.randint(y1, y2)))
            
        vertices = np.array([vertex], dtype = np.int32)
        vertices_list.append(vertices)
        
    return vertices_list


# рисует тени на изображении
def shadow_process(image, no_of_shadows, rectangular_roi, shadow_dimension):
    
    image_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS) 
    mask = np.zeros_like(image) 
    vertices_list = generate_shadow_coordinates(no_of_shadows, rectangular_roi, shadow_dimension) 
    
    for vertices in vertices_list: 
        cv2.fillPoly(mask, vertices, 255) 
    image_HLS[:, :, 1][mask[:, :, 0] == 255] = image_HLS[:,:,1][mask[:, :, 0] == 255] * 0.5 
    image_BGR = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR) 
    return image_BGR


# генерируются случайные линии указанной толщины,
# длины и наклона для дождя
def generate_random_lines(imshape, slant, drop_length, rain_type):
    
    drops = []
    area = imshape[0] * imshape[1]
    no_of_drops = area // 600

    if rain_type.lower() == 'drizzle':
        no_of_drops = area // 770
        drop_length = 10
    elif rain_type.lower() == 'heavy':
        drop_length = 30
    elif rain_type.lower() == 'torrential':
        no_of_drops = area // 500
        drop_length = 60

    for i in range(no_of_drops): 
        
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1] - slant)
            
        y = np.random.randint(0, imshape[0] - drop_length)
        drops.append((x, y))
        
    return drops, drop_length


# рисуется дождь
def rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops):
    
    for rain_drop in rain_drops:
        cv2.line(image, (rain_drop[0], rain_drop[1]), 
                 (rain_drop[0] + slant, rain_drop[1] + drop_length), drop_color, drop_width) 
        
    image = cv2.blur(image, (7, 7)) 
    brightness_coefficient = 0.7 
    
    image_HLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    image_HLS[:,:,1] = image_HLS[:, :, 1] * brightness_coefficient 
    image_BGR = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR)
    
    return image_BGR


# рисуется туман
def add_blur(image, x, y, hw, fog_coeff):
    
    overlay= image.copy()
    alpha = 0.08 * fog_coeff
    rad = hw // 2
    point = (x + hw // 2, y + hw // 2)
    
    cv2.circle(overlay, point, int(rad), (255, 255, 255), -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    return image


#  генерируются случайные координаты для тумана
def generate_random_blur_coordinates(imshape, hw):
    
    blur_points = []
    midx = imshape[1] // 2 - 2 * hw
    midy = imshape[0] // 2 - hw
    index = 1
    
    while(midx > -hw or midy > -hw):
        
        for i in range(hw // 10 * index):
            x = np.random.randint(midx, imshape[1] - midx - hw)
            y = np.random.randint(midy, imshape[0] - midy - hw)
            blur_points.append((x, y))
            
        midx -= 3 * hw * imshape[1] // sum(imshape)
        midy -= 3 * hw * imshape[0] // sum(imshape)
        index += 1
        
    return blur_points


# ===== Основные функции =====


# делает изображение ярче на указанный коэффициент от 0 до 1
# прибавляется умноженный на коэффициент параметр L из формата HLS
# если передать -1, то будет сгенерирован случайный коэффициент
def brighten(image, brightness_coeff = -1):
    
    if brightness_coeff != -1 and (brightness_coeff < 0.0 or brightness_coeff > 1.0):
        raise Exception("Brightness coefficient can only be between 0.0 to 1.0")
        
    if(brightness_coeff == -1):
        brightness_coeff_t = 1 + random.uniform(0,1) 
    else:
        brightness_coeff_t = 1 + brightness_coeff 
    image_BGR = change_light(image, brightness_coeff_t)
    return image_BGR


# делает изображение темнее на указанный коэффициент от 0 до 1
# отнимается умноженный на коэффициент параметр L из формата HLS
# если передать -1, то будет сгенерирован случайный коэффициент
def darken(image, darkness_coeff = -1): 
    
    if darkness_coeff != -1 and (darkness_coeff < 0.0 or darkness_coeff > 1.0):
        raise Exception("Darkness coefficient can only be between 0.0 to 1.0")
        
    if(darkness_coeff==-1):
        darkness_coeff_t=1- random.uniform(0,1)
    else:
        darkness_coeff_t=1- darkness_coeff  
        
    image_BGR = change_light(image, darkness_coeff_t)
    return image_BGR


# Изменят яркость на случайную 
# от нулевой до 2 раза увеличенной
def random_brightness(image):
    
    random_brightness_coefficient = 2 * np.random.uniform(0,1) 
    image_BGR = change_light(image,random_brightness_coefficient)
    return image_BGR


# добавляет тени-многоугольники на изображение
# no_of_shadows - число теней
# rectangular_roi - квадрат области, в которой будут добавлены тени
# shadow_dimension - число углов у тени
def add_shadow(image, no_of_shadows = 1, rectangular_roi = (-1,-1,-1,-1), shadow_dimension = 5):
    
    (x1, y1, x2, y2) = rectangular_roi
    if rectangular_roi == (-1,-1,-1,-1):
        x1 = 0
        y1 = 0
        x2 = image.shape[1]
        y2 = image.shape[0]
      
    elif x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x2 <= x1 or y2 <= y1:
        raise Exception("Rectangular ROI dimensions are not valid")
  
    image_BGR = shadow_process(image, no_of_shadows, (x1, y1, x2, y2), shadow_dimension)
    return image_BGR


# добавляет эффект дождя на изображение с помощью
# добавления темных линий разной длинны
# slant - угол наклона дождя в градусах (для -1 случайный от -10 до 10 включительно)
# drop_length - длинна капли (линии) в пикселях 
# drop_width - ширина капли (линии) в пикселях 
# drop_color - цвет капли по умолчанию темно-серый
# rain_type - тип дождя ('drizzle','heavy','torrential','None')
# Для 'None' все параметры регулируются в ручную
# остальные типы ращличаются плотностью дождя, длинной капель (скоростью)
def add_rain(image, slant =- 1, drop_length = 20, drop_width = 1, drop_color = (200, 200, 200), rain_type = 'None'):
    
    slant_extreme = slant
    imshape = image.shape
    
    if slant_extreme == -1:
        slant = np.random.randint(-10, 10) 
        
    rain_drops, drop_length = generate_random_lines(imshape, slant, drop_length, rain_type)
    image_BGR = rain_process(image, slant_extreme, drop_length, drop_color, drop_width, rain_drops)
    
    return image_BGR


# добавляет эффект тумана с разным коэффициентом от 0 до 1
# если -1, то случайных коэффициент
def add_fog(image, fog_coeff=-1):
    
    if fog_coeff != -1:
        if (fog_coeff < 0.0 or fog_coeff > 1.0):
            raise Exception("Fog coefficientcan only be between 0 and 1")
            
    else:
        fog_coeff = random.uniform(0.3,1)
        
    
    imshape = image.shape
    hw = int(imshape[1] // 3 * fog_coeff)
    haze_list = generate_random_blur_coordinates(imshape, hw)
    for haze_points in haze_list: 
        image = add_blur(image, haze_points[0], haze_points[1], hw, fog_coeff) 

    image_BGR = cv2.blur(image, (hw // 10, hw // 10))

    return image_BGR






