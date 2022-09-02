from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox,DISABLED,NORMAL
# import pymysql
import datetime
from functools import partial
from PIL import Image, ImageTk
import cv2
#from test import testing
import time
title="Aerial Image Classification and Person Detection"
path1="sample.jpg"
path2="sample1.jpg"
main_color='#271745'  

def logcheck():
     global username_var,pass_var
     showcheck()   

# show home page
def showhome():
    top.config(menu=menubar)
    global f
    f.pack_forget()
    f=Frame(top)
    f.config(bg=main_color)
    f.pack(side="top", fill="both", expand=True,padx=10,pady=10)
    image = Image.open("leaf.jpg")
    photo = ImageTk.PhotoImage(image.resize((top.winfo_width(), top.winfo_height()), Image.ANTIALIAS))
    label = Label(f, image=photo, bg=main_color)
    label.image = photo
    label.pack()

    l=Label(f,text="Welcome",font = "Verdana 60 bold",fg="White",bg=main_color)
    l.place(x=500,y=300)

def showcheck():
    top.title(title)
    top.config(menu=menubar)
    global f
    f.pack_forget()
    f=Frame(top)
    f.config(bg=main_color)
    f.pack(side="top", fill="both", expand=True,padx=10,pady=10)

    f1=Frame(f)
    f1.pack_propagate(False)
    f1.config(bg="white",width=500)
    f1.pack(side="left",fill="both")

    global f2
    f2=Frame(f)
    f2.pack_propagate(False)
    f2.config(bg="white",width=500)
    f2.pack(side="right",fill="both")

    f3=Frame(f)
    f3.pack_propagate(False)
    f3.config(bg=main_color,width=600)
    f3.pack(side="right",fill="both")

    f4=Frame(f3)
    f4.pack_propagate(False)
    f4.config(bg=main_color,height=200)
    f4.pack(side="bottom",fill="both")

    f7=Frame(f3)
    f7.pack_propagate(False)
    f7.config(height=20)
    f7.pack(side="top",fill="both",padx="3")

    l2=Label(f7,text="Process",font="Helvetica 13 bold")
    l2.pack()

    global lb1
    b2=Button(f4,text="Classification",font="Verdana 10 bold",command=lambda:process1(path1,lb1))
    b2.pack(pady=2)
    b3=Button(f4,text="Person Detection",font="Verdana 10 bold",command=lambda:process2(path1,lb1))
    b3.pack(pady=2)

    f5=Frame(f1)
    f5.config(bg="red")
    f5.pack(side="top",fill="both")
    
    global f6
    f6=Frame(f2)
    f6.config(bg="red")
    f6.pack(side="top",fill="both")
    l1=Label(f6,text="Result",font="Helvetica 13 bold")
    l1.pack(side="bottom",fill="both")

    global path1
    try:
        image = Image.open(path1)
    except:
        path1="sample.jpg"
        image = Image.open(path1)
        messagebox.showerror("Not an image","Choose an image") 
    photo = ImageTk.PhotoImage(image.resize((500, 350), Image.ANTIALIAS))
    label = Label(f5, image=photo, bg=main_color)
    label.image = photo
    label.pack()

    b1= Button(f1,text="Upload",command=upload)
    b1.pack(side="top", fill="both",pady=5,padx=10)

    global path2
    image = Image.open(path2)
    photo = ImageTk.PhotoImage(image.resize((500, 350), Image.ANTIALIAS))
    label = Label(f6, image=photo, bg=main_color)
    label.image = photo
    label.pack()
    
    
    lb1=Listbox(f3,width=400,height=400,font="Helvetica 13 bold")
    lb1.pack(pady=10,padx=5)
    

    
def upload():
    global path1
    path1=askopenfilename()
    showcheck()

from predict import predict1
from yolo import work
import numpy as np

def process1(path2,lb1):
    global f6,f2,top
    
    t=1
    lb1.after(t,delayed_insert,lb1,t,'Loading image..')
    lb1.update()
    t+=1
    lb1.after(t,delayed_insert,lb1,t,'Preprocessing..')
    lb1.update()
    t+=1
    lb1.after(t,delayed_insert,lb1,t,'Load model..')
    lb1.update()
    t+=1
    lb1.after(t,delayed_insert,lb1,t,'Prediction..')
    lb1.update()
    
    t+=1

    res=predict1(path2)
    work(path2)
    

    f6.pack_forget()
    f6=Frame(f2)
    f6.config(bg="white", height=350)
    f6.pack(side="top",fill="both")
    f6.pack_propagate(False)
    global lb2
    lb2=Listbox(f2,width=200,height=200,font="Helvetica 13 bold")
    lb2.pack(side="top",pady=10,padx=5)
    
    l1=Label(f6,text="Result",font="Helvetica 13 bold")
    l1.pack(side="bottom",fill="both")
    lb1.after(10,delayed_display)
    lb2.after(10,showresult,res)
    
def process2(path2,lb1):
    global f6,f2,top
    
    t=1
    lb1.after(t,delayed_insert,lb1,t,'Loading image..')
    lb1.update()
    t+=1
    lb1.after(t,delayed_insert,lb1,t,'Preprocessing..')
    lb1.update()
    t+=1
    lb1.after(t,delayed_insert,lb1,t,'Load model..')
    lb1.update()
    t+=1
    lb1.after(t,delayed_insert,lb1,t,'Prediction..')
    lb1.update()
    
    t+=1

    
    work(path2)
    

    f6.pack_forget()
    f6=Frame(f2)
    f6.config(bg="white", height=350)
    f6.pack(side="top",fill="both")
    f6.pack_propagate(False)
    global lb2
    lb2=Listbox(f2,width=200,height=200,font="Helvetica 13 bold")
    lb2.pack(side="top",pady=10,padx=5)
    
    l1=Label(f6,text="Result",font="Helvetica 13 bold")
    l1.pack(side="bottom",fill="both")
    lb1.after(10,delayed_display)
   
    
    
    
def showresult(res):
    global lb2
    lb2.insert(0,res)




def delayed_display():
    global f6,f2
    image = Image.open("res.jpg")
    photo = ImageTk.PhotoImage(image.resize((500, 350), Image.ANTIALIAS))
    label = Label(f6, image=photo, bg=main_color)
    label.image = photo
    label.pack()


def delayed_insert(label,index,message):
    label.insert(index,message)  




    




   


if __name__=="__main__":

    top = Tk()  
    top.title("Login")
    top.geometry("1900x700")
    footer = Frame(top, bg='grey', height=30)
    footer.pack(fill='both', side='bottom')

    lab1=Label(footer,text="Developed by ###",font = "Verdana 8 bold",fg="white",bg="grey")
    lab1.pack()

    menubar = Menu(top)  
    menubar.add_command(label="Home",command=showhome)  
    menubar.add_command(label="Check",command=showcheck)

    top.mainloop() 
