import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np

from keras.models import load_model

#loading the trained model to classify the images
while(True):
    model_choice = input("Would you like the 10 epoch model or the 25? (Type '10' or '25'): ")
    if(model_choice == "10"):
        model = load_model("model1_cifar_10epoch.h5")
        break
    elif(model_choice == "25"):
        model = load_model("model1_cifar_25epoch.h5")
        break
    else:
        print("Please enter in a valid choice")

#dictionary to label all the CIFAR-10 dataset classes
classes = {0:"Airplane", 1:"Car", 2:"Bird", 3:"Cat", 4:"Deer", 5:"Dog", 6:"Frog",
           7:"Horse", 8:"Ship", 9:"Truck"}

#initializing the GUI
top = tk.Tk()
top.geometry("800x600")
top.title("Image Classicication")
top.configure(background="#CDCDCD")
label = Label(top, background = "#CDCDCD", font = ("arial", 14, "bold"))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32,32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict_classes([image])[0]
    sign = classes[pred]
    print(sign)
    label.configure(foreground="#011638", text=sign)

def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path),
                        padx=10, pady=5)
    classify_b.configure(background="#364156", foreground="white", font=("arial", 10, "bold"))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25),
                            (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload = Button(top, text="Upload an image", command=upload_image,
                padx=10, pady=5)
upload.configure(background='#364156', foreground='white',
                 font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Image Classification CIFAR10", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()