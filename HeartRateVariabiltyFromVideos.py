import numpy as np
from matplotlib import pyplot as plt
import cv2
import io
import time

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# To capture video from webcam.
#cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#cap.set(cv2.CAP_PROP_FPS, 60)

# To use a video file as input
cap = cv2.VideoCapture(r'C:\Users\USER\Pictures\Picasa\Captured Videos\video1.wmv')

heartbeat_count = 128
heartbeat_values = [0]*heartbeat_count
heartbeat_times = [time.time()]*heartbeat_count
# Matplotlib graph surface
fig = plt.figure()
ax = fig.add_subplot(111)

#Performing face detection and selecting ROI as forehead
while(True):
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gray = cv2.imread('img', 0)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around face and forehead
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(img, (x+ 30, y+ 10), (x+ 60, y +20), (0, 255, 0), 2)

    x,y,w,h = faces[0]

    #calculation of heartratefrom signals from ROI
    crop_img = img[y:y + h, x:x + w]

    
    # Update the data
    heartbeat_values = heartbeat_values[1:] + [np.average(crop_img)]
    heartbeat_times = heartbeat_times[1:] + [time.time()]
    
    # Draw matplotlib graph to numpy array
    ax.plot(heartbeat_times, heartbeat_values)
    fig.canvas.draw()
    plot_img_np = np.fromstring(fig.canvas.tostring_rgb(),dtype=np.uint8, sep='')
    plot_img_np = plot_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.cla()


    #For displaying heartbeat values in the graph
    #for (x, y, w, h) in faces:
        #hr = heartbeat_values[0]
        #BPM = "HeartRate (in Bpm) : "
        #BPM = "{}{} ".format(BPM, hr)
        #cv2.putText(plot_img_np, BPM, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,255,0),2)
    
    #Display the Heart rate Graph
    cv2.imshow('Graph', plot_img_np)
    plt.xlabel("Instant")
    plt.ylabel("Heart Rate Variabilty")

    #Display video with bounded box
    cv2.imshow('img', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
       
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
