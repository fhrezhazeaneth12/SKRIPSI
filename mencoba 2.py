# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import tkinter as tk
from tkinter import Tk, Canvas, Frame, BOTH
import cv2
import PIL.Image, PIL.ImageTk
import time
import datetime as dt
import argparse
import pyautogui
from tk import *
from PIL import ImageTk, Image


gui = tk.Tk ()
gui.configure(bg='white')
gui.title ("Kalsifikasi Wajah")

#menentukan ukuran window
gui.geometry("700x700")

#membuat judul aplikasi
label1 = tk.Label (gui,text="Pengenalan Wajah Pecandu Narkoba \n Dengan Metode Viola Jones dan Fisherface", font=("Times New Roman",22), fg='blue', bg='white')


label10 =tk.Label(gui, text= "Hasil Foto", font=("Times New Roman",14), fg='blue', bg='white')
label10.place(x=25,y=100)

label11 = tk.Label (gui, text= "Hasil Pengenalan: ", font=("Times New Roman",14), fg='blue', bg='white')
label11.place(x=400,y=250)

label12 = tk.Label (gui, text= "Confidence: ", font=("Times New Roman",14), fg='blue', bg='white')
label12.place(x=400,y=300)

#meletakkan logo upn
B12 = Image.open("D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\Logo UPN.jpg")
B13 = B12.resize((100, 100))
B14 = ImageTk.PhotoImage(B13)
label21 = tk.Label(image=B14)
label21.image = B14
label21.place(x= 500, y = 110)

#membuat perintah ambil gambar
def ambil_gambar():
    camera = cv2.VideoCapture(0)
    for i in range(1):
        return_value, image = camera.read()
        cv2.imwrite('opencv' + str(i) + '.png', image)
    del (camera)

    # Create a photoimage object of the image in the path
    image = Image.open("D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\opencv0.png")
    img=image.resize((300, 250))
    my_img = ImageTk.PhotoImage(img)
    label2 = tk.Label(image=my_img)
    label2.image = my_img

    # Position image
    label2.place(x= 15, y = 140)

#membuat perintah button analysis fisher face
def analyse_fisherface():
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import re

    cascPathface = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    cascPatheyes = os.path.dirname(
        cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"
    eyeCascade = cv2.CascadeClassifier(cascPatheyes)
    faceCascade = cv2.CascadeClassifier(cascPathface)

#meletakkan data latih INI NANTI KAMU RUBAH DENGAN DATA LATIH PECANDU DAN BUKAN PECANDU
#NANTI KAMU KASIH FOLDER DENGAN FILE PECANDU1.png. PECANDU2.pndg BUKAN PECANDU1.png dst dst dst
    face_db = [
        "D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\Dataset\Bukan Pecandu\BUKAN PECANDU 1.png",
        "D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\Dataset\Bukan Pecandu\BUKAN PECANDU 2.png",
        "D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\Dataset\Bukan Pecandu\BUKAN PECANDU 3.png",
        "D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\Dataset\Bukan Pecandu\BUKAN PECANDU 4.png",
        "D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\Dataset\Bukan Pecandu\BUKAN PECANDU 5.png",
        "D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\Dataset\Bukan Pecandu\BUKAN PECANDU 6.png",
        "D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\Dataset\Bukan Pecandu\BUKAN PECANDU 7.png",
        "D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\Dataset\Bukan Pecandu\BUKAN PECANDU 8.png",
        "D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\Dataset\Bukan Pecandu\BUKAN PECANDU 9.png",
        "D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\Dataset\Bukan Pecandu\BUKAN PECANDU 10.png",
    ]

    #membuat perintah button deteksi wajah VIOLA JONES UNTUK DATA LATIH
    def detect_face(img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(60, 60),
                                             flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            faceROI = img [y:y + h, x:x + w]

        return gray

    # tampilkan wajah
    faces = [];
    ids = []
    index = 0
    for img_path in face_db:
        # print(img_path
        detected_face = detect_face(img_path)
        # plt.imshow(detected_face)
        # plt.show()
        faces.append(detected_face)
        ids.append(index)

        index = 1

    ids = np.array(ids)

    # TRAINING
    model = cv2.face.FisherFaceRecognizer_create()
    model.train(faces, ids)
    model.save("model.yml")

    # MENCARI DATA TEST. INI KAMU UBAH AJA KECUALI NAMA IMAGER NYA (opencv0.png)
    target_path = "D:\KULIAH\Fhrezha\SKRIPSI\BISMILLAH\Pak Bayu\FIX\opencv0.png"

    #MENERAAPKAN VIOLA JONES PADA DATA TEST
    target = detect_face(target_path)

    #MENGHITUNG TINGKAT KECOCOKAN
    idx, confidence = model.predict(target)

    A = face_db[idx]
    B = detect_face(A)
    C = (confidence)
    D = round (C,2)

#membuat output text sesuai dengan nama data latih (pecandu/bukan pecandu)
    string = A
    FFFF = string[66:100]
    pattern = r'[0-9]'
    new_string = re.sub(pattern, '', FFFF)
    remove_type = new_string[0:-4]
    label2 = tk.Label(gui, text= remove_type,
                      font=("Times New Roman", 14),fg='black',bg='white')
    label2.place (x=537, y=250)
    label3 = tk.Label(gui, text= D,
                      font=("Times New Roman", 14),fg='black',bg='white')
    label3.place(x=493, y=300)

def reset ():
    label4 = tk.Label(gui, text= ".................",
                      font=("Times New Roman", 14),fg='white',bg='white')
    label4.place (x=537, y=250)


    label5 = tk.Label(gui, text= ".................",
                      font=("Times New Roman", 14),fg='white',bg='white')
    label5.place(x=493, y=300)

    canvas = Canvas(gui, width=300, height=250, bg='white')
    canvas.place(x=15, y=140)

Button1 = tk.Button (gui, text="Ambil \n Gambar", command=ambil_gambar)
Button1.place(x=50, y=600)

Button2 =tk.Button(gui, text="process \n fisherface", command=analyse_fisherface)
Button2.place (x=120, y= 600)

Button3 = tk.Button (gui, text = "tutup aplikasi", command=gui.quit)
Button3.place (x=600, y=600)

Button4 = tk.Button(gui, text="reset", command=reset)
Button4.place (x=550, y=600)


def nyalakan_kamera():
    # Create a Label to capture the Video frames
    label = tk.Label(gui)
    cap = cv2.VideoCapture(0)

    # Define function to show frame
    def show_frames():
        # Get the latest frame and convert into Image
        cv2image = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB)
        img1 = Image.fromarray(cv2image)
        img2 = img1.resize((300, 250))
        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image=img2)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        # Repeat after an interval to capture continiously
        label.after(20, show_frames)
        label.place(x=15, y=140)

    show_frames()
    gui.mainloop()


Button5 = tk.Button(gui, text="nyalakan kamera", command=nyalakan_kamera)
Button5.place(x=550, y=400)

#menampilkan gambar di GUI

label1.pack()
canvas=Canvas(gui, width=330, height=280, bg='white')
canvas.place (x=0,y=125)
canvas.create_rectangle(10, 10, 325, 275, fill='')
gui.mainloop()
