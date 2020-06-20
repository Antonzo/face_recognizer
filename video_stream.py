from threading import Thread
import cv2


# класс для оптимизации видеопотока opencv
class VideoStream:
    
	def __init__(self, src=0, name='Video stream'):
        
		# инициализация потока и считывание первого фрейма
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		# инициализации имени потока
		self.name = name

		# нужно ли останавливать поток
		self.stopped = False


    # запускает поток для чтения кадров из видеопотока
	def start(self):
        
        # инициализация потока с вызываемой функцией self.update в run()
		t = Thread(target=self.update, name=self.name, args=())
		t.daemon = True
        
        # организует вызов метода run() в отдельном потоке 
		t.start()
		return self

    
    # считывание фреймов, пока поток не остановится
	def update(self):
		
		while True:
            
			if self.stopped:
				return

			(self.grabbed, self.frame) = self.stream.read()
            
    
    # текущий фрейм
	def read(self):
		return self.frame


    # остановка видеопотока
	def stop(self):
		self.stopped = True