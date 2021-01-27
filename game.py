import pygame 
from PIL import Image
import cv2
from resizeimage import resizeimage
import onnx
import onnxruntime
import numpy as np
import time
import json

model_path = "./model.onnx"
pygame.init()
    
dimension = (300,300)
radius=10
black=(0,0,0)
white=(255,255,255)

window = pygame.display.set_mode((dimension))
pygame.display.set_caption('Digit Recognizer')

def predict(img):
   session = onnxruntime.InferenceSession(model_path)
   input_name = session.get_inputs()[0].name
   output_name = session.get_outputs()[0].name
   img = img.reshape((1,1,28,28))
   data = json.dumps({'data':img.tolist()})
   data = np.array(json.loads(data)['data']).astype('float32')
   result = session.run([output_name],{input_name:data})
   prediction = int(np.argmax(np.array(result).squeeze(),axis=0))
   return prediction

def digPredict(window):
    data = pygame.image.tostring(window,'RGBA')
    img = Image.frombytes('RGBA',(dimension),data)
    img = resizeimage.resize_cover(img,[28,28])
    imgArray = np.asarray(img)
    imgArray = cv2.cvtColor(imgArray,cv2.COLOR_RGB2GRAY)
    _,imgArray = cv2.threshold(imgArray,130,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    imgArray = imgArray/255
    prediction = predict(imgArray)
    return prediction
    
def textObjects(text,font):
    textSurface = font.render(text,True,white)
    return textSurface, textSurface.get_rect()

def display(text,x,y,size):
    largeText = pygame.font.Font('freesansbold.ttf',size)
    TextSurf,TextRec = textObjects(text,largeText)
    TextRec.center = (x,y)
    window.blit(TextSurf,TextRec)
    pygame.display.update()

def start():
    run = False
    window.fill(black)
    pygame.display.flip()
    tick = 0
    tock = 0
    startDraw = False
    while not run:
        if tock - tick >= 2 and startDraw:
            predVal = digPredict(window)
            window.fill(black)
            display("Predicted Value: "+str(predVal), 150, 150, 20)
            time.sleep(2)#sleep for 5 seconds
            window.fill(black)
            pygame.display.flip()
            tick = 0
            tock = 0
            startDraw = False
            continue

        tock = time.process_time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = True
        if pygame.mouse.get_pressed()[0]:
            spot = pygame.mouse.get_pos()
            pygame.draw.circle(window,white,spot,radius)
            pygame.display.flip()
            tick = time.process_time()
            startDraw = True

if __name__ == "__main__":
	start()
	pygame.quit()
