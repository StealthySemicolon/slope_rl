import time

import cv2
import mss
import numpy


with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 112, "left": 1, "width": 800, "height": 600}

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = cv2.cvtColor(numpy.array(sct.grab(monitor)), cv2.COLOR_RGBA2RGB)
        
        img = cv2.resize(img, (200, 150))
        img = cv2.resize(img, (800, 600))
        # Display the picture
        cv2.imshow("OpenCV/Numpy normal", img)

        # Display the picture in grayscale
        # cv2.imshow('OpenCV/Numpy grayscale',
        #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))


        # Press "q" to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break