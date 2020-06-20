import datetime


# класс для вычисления fps
class FPS:
    
	def __init__(self):
        # хранит время начала, время окончания и общее количество кадров
		self._start = None
		self._end = None
		self._numFrames = 0

    
    # начало отсчета таймера
	def start(self):
		self._start = datetime.datetime.now()
		return self


    # конец отсчета таймера
	def stop(self):
		self._end = datetime.datetime.now()
		return self

	
    # инкрементирование числа фреймов
	def update(self):
		self._numFrames += 1
		return self

	
    # возвращает суммарное число секунд
	def elapsed(self):
		return (self._end - self._start).total_seconds()

	
    # вычисляет число фреймов секунду
	def fps(self):
		return self._numFrames / self.elapsed()