import argparse
import sys
import os
import json
import requests
import cv2


# файл со ссылками 
dataturks_JSON_FilePath = ''
# путь к директории, куда будут сохраняться изображения
image_download_dir = ''
# путь к директории, куда будут сохраняться XML файлы
pascal_voc_xml_dir = ''


# загружает изображение, если его еще не существует
# возвращает путь к файлу 
# image_url - url изображения
# image_dir - директория, в которую будет сохранена web-страница
# с изображением
def maybe_download(image_url, image_dir):
    
    image_url = image_url.replace('/', os.path.sep)
    
    file_name = image_url.split(os.path.sep)[-1]
    file_path = os.path.join(image_dir, file_name)
    
    if (os.path.exists(file_path)):
        return file_path

    # попытка загрузить изображение
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                f.write(response.content)
                return file_path
        else:
            raise ValueError('Not a 200 response')
    
    # ошибка, когда не удалось загрузить изображение по заданному url
    except Exception as e:
        raise e


def xml_for_obj(obj_label, obj_data, width, height):

    if len(obj_data['points']) == 4:
        
        # обычно 4 точки прямугольника
        x_min = width * min(obj_data['points'][0][0], obj_data['points'][1][0], obj_data['points'][2][0], obj_data['points'][3][0])
        y_min = height * min(obj_data['points'][0][1], obj_data['points'][1][1], obj_data['points'][2][1],
                           obj_data['points'][3][1])

        x_max = width * max(obj_data['points'][0][0], obj_data['points'][1][0], obj_data['points'][2][0],
                           obj_data['points'][3][0])
        y_max = height * max(obj_data['points'][0][1], obj_data['points'][1][1], obj_data['points'][2][1],
                           obj_data['points'][3][1])

    else:
        
        # но иногда в виде двух точек
        # левый верхний - '0',  правый верхний  '1'
        x_min = int(obj_data['points'][0]['x'] * width)
        y_min = int(obj_data['points'][0]['y'] * height)
        x_max = int(obj_data['points'][1]['x'] * width)
        y_max = int(obj_data['points'][1]['y'] * height)

    xml = "<object>\n"
    xml += "\t<name>" + obj_label + "</name>\n"
    xml += "\t<pose>Unspecified</pose>\n"
    xml += "\t<truncated>Unspecified</truncated>\n"
    xml += "\t<difficult>Unspecified</difficult>\n"
    xml += "\t<occluded>Unspecified</occluded>\n"
    xml += "\t<bndbox>\n"
    xml += "\t\t<xmin>" + str(x_min) + "</xmin>\n"
    xml += "\t\t<xmax>" + str(x_max) + "</xmax>\n"
    xml += "\t\t<ymin>" + str(y_min) + "</ymin>\n"
    xml += "\t\t<ymax>" + str(y_max) + "</ymax>\n"
    xml += "\t</bndbox>\n"
    xml += "</object>\n"
    return xml


# конвертирует изображение в формат pascalVOC
# возвращает True/False при удачной/неудачной конвертации
# dataturks_labeled_item - одно размеченное изображение из БД dataturks в формате JSON
# image_dir - путь, куда загружать изображения
# xml_out_dir - путь, куда сохранять итоговые XML файлы
def convert_to_PascalVOC(dataturks_labeled_item, image_dir, xml_out_dir):

    image_dir = image_dir.replace('/', os.path.sep)
    xml_out_dir = xml_out_dir.replace('/', os.path.sep)
    # попытка загрузить JSON-файла
    try:
        data = json.loads(dataturks_labeled_item)
        if len(data['annotation']) == 0:
            return False;
        
        # получение ширины, длины и ссылки на изображение
        width = data['annotation'][0]['imageWidth']
        height = data['annotation'][0]['imageHeight']
        image_url = data['content']
        
        # загрузка изображения
        file_path = maybe_download(image_url, image_dir)
        
        img = cv2.imread(file_path)
        height, width, _ = img.shape

        file_name = file_path.split(os.path.sep)[-1]
        folder_name = image_dir.split(os.path.sep)[-1]

        # формирование XML файла 
        xml = "<annotation>\n<folder>" + folder_name + "</folder>\n"
        xml += "<filename>" + file_name +"</filename>\n"
        xml += "<path>" + file_path +"</path>\n"
        xml += "<source>\n\t<database>Unknown</database>\n</source>\n"
        xml += "<size>\n"
        xml += "\t<width>" + str(width) + "</width>\n"
        xml += "\t<height>" + str(height) + "</height>\n"
        xml += "\t<depth>Unspecified</depth>\n"
        xml += "</size>\n"
        xml += "<segmented>Unspecified</segmented>\n"
        
        for obj in data['annotation']:
            
            if not obj:
                continue;
                
            # PascalVOC поддерживает только прямоугольники
            if "shape" in obj and obj["shape"] != "rectangle":
                continue;
            
            # создание списка меток
            obj_labels = obj['label']
            if obj_labels is not list:
                obj_labels = [obj_labels]

            for obj_label in obj_labels:
                xml = xml + xml_for_obj(obj_label, obj, width, height)

        xml = xml + "</annotation>"

        # запись в файл
        xmlFilePath = os.path.join(xml_out_dir, file_name + ".xml")
        with open(xmlFilePath, 'w') as f:
            f.write(xml)
            
        return True
    except Exception as e:
        return False


def main():

    # считываем JSON файл со ссылками
    lines = []
    with open(dataturks_JSON_FilePath, 'r') as f:
        lines = f.readlines()

    if (not lines or len(lines) == 0):
        return

    # для каждой ссылки конвертируем
    count = 0;
    success = 0
    for line in lines:
        status = convert_to_PascalVOC(line, image_download_dir, pascal_voc_xml_dir)
        if (status):
            success = success + 1

        count += 1
        
    # коэф. успеха  
    # print(success / count)


main()