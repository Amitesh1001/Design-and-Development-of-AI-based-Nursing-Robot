import pygame
import time
from gtts import gTTS
from mutagen.mp3 import MP3
import time
import telepot
bot = telepot.Bot('6194426199:AAFmjeYwM2ITzGCBo0MR1oJ_ydxHyzIDjGs')

def text_to_speech(text1):
    myobj = gTTS(text=text1, lang='en-us', tld='com', slow=False)
    myobj.save("voice.mp3")
    print('\n------------Playing--------------\n')
    song = MP3("voice.mp3")
    pygame.mixer.init()
    pygame.mixer.music.load('voice.mp3')
    pygame.mixer.music.play()
    time.sleep(song.info.length)
    pygame.quit()

import numpy as np
import os
import time
import RPi.GPIO as GPIO
import time
import RPi.GPIO as GPIO
import spidev
import os
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

#MOVEMENT
IN1=21
IN2=20
IN3=12
IN4=16

GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

GPIO.output(IN1, False)
GPIO.output(IN2, False)
GPIO.output(IN3, False)
GPIO.output(IN4, False)

#DOOR ONE
IN5=23
IN6=24
#DOOR TWO
IN7=27
IN8=22
#DOOR THREE
IN9=3
IN10=2

GPIO.setup(IN5, GPIO.OUT)
GPIO.setup(IN6, GPIO.OUT)
GPIO.setup(IN7, GPIO.OUT)
GPIO.setup(IN8, GPIO.OUT)
GPIO.setup(IN9, GPIO.OUT)
GPIO.setup(IN10, GPIO.OUT)

GPIO.output(IN5, False)
GPIO.output(IN6, False)
GPIO.output(IN7, False)
GPIO.output(IN8, False)
GPIO.output(IN9, False)
GPIO.output(IN10, False)

def FORWORD(t):
    print('FORWORD')
    GPIO.output(IN1,False)
    GPIO.output(IN2, True)
    GPIO.output(IN3,False)
    GPIO.output(IN4,True)
    time.sleep(t)
    
def BACKWORD(t):
    print('FORWORD')
    GPIO.output(IN1,True)
    GPIO.output(IN2, False)
    GPIO.output(IN3,True)
    GPIO.output(IN4,False)
    time.sleep(t)
    
def STOP():
    print('STOP')
    GPIO.output(IN1, False)
    GPIO.output(IN2, False)
    GPIO.output(IN3, False)
    GPIO.output(IN4, False)
    time.sleep(1)
    
def RIGHT(t):
    print('RIGHT')
    GPIO.output(IN1,False)
    GPIO.output(IN2, True)
    GPIO.output(IN3,True)
    GPIO.output(IN4,False)
    time.sleep(t)
    
    
def LEFT(t):
    print('LEFT')
    GPIO.output(IN1,True)
    GPIO.output(IN2, False)
    GPIO.output(IN3,False)
    GPIO.output(IN4,True)
    GPIO.output(IN7, False)
    GPIO.output(IN8, False)
    time.sleep(t)

def ONE(Id):
    print ("ONE")
    GPIO.output(IN5,True)
    GPIO.output(IN6, False)
    GPIO.output(IN7, False)
    GPIO.output(IN8, False)
    time.sleep(0.5)
    GPIO.output(IN5,False)
    GPIO.output(IN6, False)
    time.sleep(1)
    
    print('{} take your medicine'.format(Id))
    text_to_speech('{} take your medicine'.format(Id))
    
    time.sleep(2)
    GPIO.output(IN5,False)
    GPIO.output(IN6, True)
    time.sleep(0.5)
    GPIO.output(IN5,False)
    GPIO.output(IN6, False)
    time.sleep(1)
    
    GPIO.output(relay,True)
    time.sleep(1)
    GPIO.output(relay,False)
    time.sleep(1)
    
def TWO(Id):
    print ("TWO")
    GPIO.output(IN7,True)
    GPIO.output(IN8, False)
    GPIO.output(IN7, False)
    GPIO.output(IN8, False)
    time.sleep(0.5)
    GPIO.output(IN7,False)
    GPIO.output(IN8, False)
    time.sleep(1)
    print('{} take your medicine'.format(Id))
    text_to_speech('{} take your medicine'.format(Id))
    
    time.sleep(2)
    GPIO.output(IN7,False)
    GPIO.output(IN8, True)
    time.sleep(0.5)
    GPIO.output(IN7,False)
    GPIO.output(IN8, False)
    time.sleep(1)
    
    GPIO.output(relay,True)
    time.sleep(1)
    GPIO.output(relay,False)
    time.sleep(1)
    
def THREE(Id):
    print ("THREE")
    GPIO.output(IN9,True)
    GPIO.output(IN10, False)
    GPIO.output(IN7, False)
    GPIO.output(IN8, False)
    time.sleep(0.5)
    GPIO.output(IN9,False)
    GPIO.output(IN10, False)
    time.sleep(1)
    print('{} take your medicine'.format(Id))
    text_to_speech('{} take your medicine'.format(Id))
    
    time.sleep(2)
    GPIO.output(IN9,False)
    GPIO.output(IN10, True)
    time.sleep(0.5)
    GPIO.output(IN9,False)
    GPIO.output(IN10, False)
    time.sleep(1)
    
    GPIO.output(relay,True)
    time.sleep(1)
    GPIO.output(relay,False)
    time.sleep(1)
    
def PATH1():
    print('PATH1')
    FORWORD(5)
    STOP()
    
def PATH2():
    print('PATH2')
    LEFT(5)
    STOP()
    FORWORD(5)
    STOP()
    
def PATH3():
    print('PATH3')
    RIGHT(5)
    STOP()
    FORWORD(5)
    STOP()

relay=6
GPIO.setup(relay,GPIO.OUT)
GPIO.output(relay,False)
SW = 19
GPIO.setup(SW, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
BZ = 26
GPIO.setup(BZ, GPIO.OUT)
GPIO.output(BZ, False)

spi = spidev.SpiDev()
spi.open(0,0)
spi.max_speed_hz=5000

def ReadChannel(channel):
  adc = spi.xfer2([1,(8+channel)<<4,0])
  data = ((adc[1]&3) << 8) + adc[2]
  return data
 
# Function to convert data to voltage level,
# rounded to specified number of decimal places.
def ConvertVolts(data,places):
  volts = (data * 3.3) / float(1023) + 25
  volts = round(volts,places)
  return volts
 
# Function to calculate temperature from
# TMP36 data, rounded to specified
# number of decimal places.
def ConvertTemp(data,places):
 
  # ADC Value
  # (approx)  Temp  Volts
  #    0      -50    0.00
  #   78      -25    0.25
  #  155        0    0.50
  #  233       25    0.75
  #  310       50    1.00
  #  465      100    1.50
  #  775      200    2.50
  # 1023      280    3.30
 
  temp = ((data * 330)/float(1023))#-50 40
  temp = round(temp,places)+25
  return temp

def SENSOR_TEMP():
    temp_level = ReadChannel(0)
    temp_volts = ConvertVolts(temp_level,2)
    temp       = ConvertTemp(temp_level,2)
    print("Temp :{} deg C".format(temp))
    time.sleep(1)
    bot.sendMessage('1721422651', str("Temp :{} deg C".format(temp)))
    text_to_speech("Temperature is {} degree celcius".format(temp))

HR_SENSOR = 13
tempFlag = 0
bpFlag = 0
hrFlag = 0
GPIO.setup(HR_SENSOR,GPIO.IN)

def HEART_BEAT():
     global tempFlag
     global bpFlag
     global hrFlag
     tempFlag = 0
     bpFlag   = 0
     hrFlag   = 1
     if hrFlag == 1 :
      print('Hold The finger On sensor')
      
      sensorCounter = 0
      startTime     = 0
      endTime       = 0
      rateTime      = 0
      while sensorCounter < 1 and  hrFlag == 1:
        if (GPIO.input(HR_SENSOR)):
          if sensorCounter == 0:
            startTime = int(round(time.time()*1000))
            #print startTime
          sensorCounter = sensorCounter + 1
          #print sensorCounter
          while(GPIO.input(HR_SENSOR)):
            if hrFlag == 0:
              break
            pass

      time.sleep(1)      
      endTime  = int(round(time.time()*1000))
      #print endTime
      rateTime = endTime - startTime
      #print rateTime
      rateTime = rateTime / sensorCounter
      heartRate = (60000 / rateTime) #/ 3 
      heartRate = abs(heartRate)
      heartRate=int(heartRate+20)
      print (heartRate)
      bot.sendMessage('1721422651', str("HeartRate is {}".format(heartRate)))
      text_to_speech("HeartRate is {}".format(heartRate))
      
# Import OpenCV2 for image processing
import cv2
import RPi.GPIO as GPIO
# Import numpy for matrices calculations
import numpy as np
import time
from datetime import datetime

# Create Local Binary Patterns Histograms for face recognization
#recognizer = cv2.face.LBPHFaceRecognizer_create() 
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

text_to_speech('nurse robot')
while True:
    #HEART_BEAT()
    if GPIO.input(SW) == False:
        print('Emergency')
        bot.sendMessage('1721422651', str("EMERGENCY"))
        GPIO.output(BZ, True)
        time.sleep(1)
        GPIO.output(BZ, False)
        time.sleep(1)
        
    now = datetime.now()
    TIME = now.strftime('%H:%M')
    print(TIME)
    time.sleep(1)
    
    if TIME == '14:01':
        counter = 0
        while True:
            counter += 1
            if counter > 3:
                break
            if counter == 1:
                PATH1()
            if counter == 2:
                PATH2()
            if counter == 3:
                PATH3()
                
            kcount = 0
            ucount = 0
            tcount =0
            
            # Initialize and start the video frame capture
            cam = cv2.VideoCapture(0)

            text_to_speech('please look into the camera')
            print('please look into the camera')
            # Loop
            while True:
                ret, image_frame =cam.read()
             
                # Convert the captured frame into grayscale
                gray = cv2.cvtColor(image_frame,cv2.COLOR_BGR2GRAY)

                # Get all face from the video frame
                faces = faceCascade.detectMultiScale(gray, 1.2,5)

                # For each face in faces
                for(x,y,w,h) in faces:
                    
                    tcount += 1
                    # Create rectangle around the face
                    cv2.rectangle(image_frame, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 2)

                    # Recognize the face belongs to which ID
                    Id,i= recognizer.predict(gray[y:y+h,x:x+w])

                    print(Id, i)
                    
                    if int(i) < 65:
                        kcount += 1
                        if(Id == 1):
                            Id = "Amithesh"
                            
                            
                        if(Id == 2):
                            Id = "Adithya"
                            

                        if(Id == 3):
                            Id = "gautham"
                           
                        
                        print(Id)
                        cv2.putText(image_frame, str(Id), (x,y-40), font, 1, (0,255,0), 2)
                        
                    else:
                        ucount += 1
                        print("Unknown")
                        cv2.imwrite("frame.png",image_frame)
                        cv2.putText(image_frame, "Unknown", (x,y-40), font, 1, (0,255,0), 1)
                        
                # Display the video frame with the bounded rectangle
                cv2.imshow('im',image_frame)
                # If 'q' is pressed, close program
                if cv2.waitKey(100) & tcount > 15:
                    break
                            
            # Stop the camera
            cam.release()
            # Close all windows
            cv2.destroyAllWindows()

            if kcount > 5:
                text_to_speech('please place your finger on sensors')
                if ID == 3:
                    bot.sendMessage('1721422651', str("Details for Gautham"))
                     
                if ID == 2:
                    bot.sendMessage('1721422651', str("Details for Adithya"))
                
                if ID == 1:
                    bot.sendMessage('1721422651', str("Details for Amitesh"))
                    
                SENSOR_TEMP()
                HEART_BEAT()
                if counter == 1:
                    ONE(Id)
                if counter == 2:
                    TWO(Id)
                if counter == 3:
                    THREE(Id)
                
            else:
                print('Face not recognised')
                text_to_speech('Face not recognised')
