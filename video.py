#change upscale factor in processing.py
#will have some latency while using upscale_factor = 3 or 4

import numpy as np
import cv2
from processing import convert_frame

cap = cv2.VideoCapture(0)
w,h = int(cap.get(3)), int(cap.get(4))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('FSRCNN',convert_frame(frame = frame, w = w, h = h))
    cv2.imshow('original',frame)
    c = cv2.waitKey(1)
    if c == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()