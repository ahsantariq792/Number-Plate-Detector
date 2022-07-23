from PIL.Image import ImageTransformHandler
import cv2
import numpy as np
import pytesseract
import random, threading, webbrowser
from flask import jsonify
from flask import Flask, send_file, request , send_from_directory

# giving path of tesseract library
pytesseract.pytesseract.tesseract_cmd = r'D:\Python\Lib\Tesseract\tesseract.exe'

cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")
app = Flask(__name__)


@app.route("/img_filename")
# main function
def extract_num(img_filename):
    img = cv2.imread(img_filename)
    # Img To Gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    nplate = cascade.detectMultiScale(gray, 1.1, 4)
    # crop portion
    for (x, y, w, h) in nplate:
        wT, hT, cT = img.shape
        a, b = (int(0.02 * wT), int(0.02 * hT))
        plate = img[y + a:y + h - a, x + b:x + w - b, :]
        # make the img more darker to identify LPR
        kernel = np.ones((1, 1), np.uint8)
        plate = cv2.dilate(plate, kernel, iterations=1)
        plate = cv2.erode(plate, kernel, iterations=1)
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        # (thresh, plate) = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)
        (thresh, plate) = cv2.threshold(plate_gray, 150, 180, cv2.THRESH_BINARY)

        # read the text on the plate
        read = pytesseract.image_to_string(plate)
        read = ''.join(e for e in read if e.isalnum())
        stat = read[0:2]
        print("Number Plate", read)

        # putting rectangle on number plate
        cv2.rectangle(img, (x, y), (x + w, y + h), (51, 51, 255), 2)
        cv2.rectangle(img, (x - 1, y - 40), (x + w + 1, y), (51, 51, 255), -1)
        cv2.putText(img, read, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # showing number plate
        cv2.imshow("plate", plate)

        # replacing result image with number plate

        cv2.imwrite("images/result.png", plate)

    # text = pytesseract.image_to_string("result.png")
    # print(text)
    # print("Number Plate : ",read)

    cv2.imshow("Result", img)
    if cv2.waitKey(0) == 113:
        exit()
    cv2.destroyAllWindows()







# calling function
extract_num("images/car0.jpg")

# port = 5000
# url = "http://127.0.0.1:{0}".format(port)
#
# threading.Timer(1.25, lambda: webbrowser.open(url)).start()
#
# app.run(port=port, debug=False)
