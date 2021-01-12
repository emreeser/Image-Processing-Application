import cv2
import numpy as np
from tkinter import *
import tkinter.messagebox
import tkinter.simpledialog
from matplotlib import pyplot as plt
from skimage.filters import gabor,roberts,median,sato,unsharp_mask,sobel,meijering,hessian,rank_order,prewitt,gaussian
from skimage.morphology import disk
from skimage.transform import swirl
import imutils
import skimage.segmentation as seg
from skimage.segmentation import active_contour
import skimage.color as color
from mpl_toolkits.axes_grid1 import ImageGrid

#using Tkinter 

app = Tk()
app.title('Image Processing Application')
app.geometry('1200x800')

label_image_entry = Label(app, text='Image Entry', font = "Verdana 13 bold", fg='blue')
label_image_entry.place(x=10, y=5)

image_text = StringVar()
part_entry = Entry(app, textvariable=image_text)
part_entry.place(x=10, y=40)
part_label1 = Label(app, text='(etc: car.jpg)', font="Times, 13")
part_label1.place(x=180, y=35)

label_video_entry = Label(app, text='Video Entry', font = "Verdana 13 bold", fg='purple')
label_video_entry.place(x=10, y=120)
video_text = StringVar()
part_entry = Entry(app, textvariable=video_text)
part_entry.place(x=10, y=155)
part_label2 = Label(app, text='(etc: video.mp4)', font="Times, 13")
part_label2.place(x=180, y=150)

def open_image():
    if image_text.get() != '':
        image1 = cv2.imread(image_text.get())
        image1 = cv2.resize(image1, (700, 500))
        cv2.imshow('Original Image(first)', image1)
open_button1 = Button(app, text='Open Image', font="Times, 10",fg='blue', command=open_image)
open_button1.place(x=30, y=75)

def save_image(image):
    result = tkinter.messagebox.askquestion('SAVE Image', 'Do you want to save the image?')
    if result == 'yes':
        result2 = tkinter.simpledialog.askstring("Name of the Image File","You should write the image file name correctly\n(etc: wolf.jpg {.png/.jpg/.jpeg})")
        if result2 is not None:
            image = cv2.convertScaleAbs(image, alpha=(255.0))
            cv2.imwrite(result2, image)
            tkinter.messagebox.showinfo('SAVED', 'Saving was completed successfully')
    else:
        tkinter.messagebox.showinfo('NOT SAVED', 'You did not save the image \n    You can continue')

def open_video():
    if video_text.get() != '':
        cap = cv2.VideoCapture(video_text.get())
        while True:
            ret, frame = cap.read()
            cv2.imshow('Original Video -- for exit press Q', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
open_button2 = Button(app, text='Open Video', font="Times, 10",fg='purple', command=open_video)
open_button2.place(x=30, y=190)



################# filters

def do_filter():
    if filvar.get() == 'prewitt':
        if image_text.get() != '':
            image = cv2.imread(image_text.get(), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prew=prewitt(image)
            prew = cv2.resize(prew, (600, 500))
            image = cv2.resize(image, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("prewitt Filter", prew)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(prew)
    if filvar.get() == 'threshold':
        if image_text.get() != '':
            image = cv2.imread(image_text.get(), 1)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(image,140,255, cv2.THRESH_BINARY)[1]
            thresh = cv2.resize(thresh, (600, 500))
            image = cv2.resize(image, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("threshold Filter", thresh)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(thresh)
    if filvar.get() == 'gabor':
        if image_text.get() != '':
            image = cv2.imread(image_text.get(), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            filt_gabor, filt_imag = gabor(image, frequency=1.48)
            filt_gabor = cv2.resize(filt_gabor, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("gabor", filt_gabor)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(filt_gabor)
    if filvar.get() == 'hessian':
        depth = cv2.CV_16S
        kernel_size = 3
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hess=hessian(image, sigmas=range(2,8,50))
            hess = cv2.resize(hess, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow('hessian', hess)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(hess)
    if filvar.get() == 'meijering':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mei=meijering(image ,sigmas=range(1, 4, 500))
            image = cv2.resize(image, (550, 500))
            mei = cv2.resize(mei, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow('meijering', mei)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(mei)
    if filvar.get() == 'sobel':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sobell=sobel(image)
            sobell = cv2.resize(sobell, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow('sobel', sobell)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(sobell)
    if filvar.get() == 'unsharp_mask':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            usmask = unsharp_mask(image, radius=20, amount=1)
            usmask = cv2.resize(usmask, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow(' unsharp_mask', usmask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(usmask)
    if filvar.get() == 'sato':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            satoo=sato(image, sigmas=range(5,11,4), black_ridges=True)
            satoo = cv2.resize(satoo, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow('sato', satoo)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(satoo)
    if filvar.get() == 'median':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            med = median(image, disk(2.3))
            med = cv2.resize(med, (550, 500))
            image = cv2.resize(image, (550, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow('median', med )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(med)
    if filvar.get() == 'roberts':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            robert=roberts(image, mask=None)
            image = cv2.resize(image, (600, 500))
            robert = cv2.resize(robert, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow('roberts', robert)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(robert)

filvar = StringVar()
filvar.set('Filtering')
filter_names = {'prewitt', 'threshold', 'gabor', 'hessian', 'meijering', 'sobel', 'unsharp_mask',
            'sato', 'median', 'roberts','! NONE !'}
filter_menu = OptionMenu(app, filvar, *filter_names)
label_f = Label(app, text="Select \na Filter",font="Times, 11",fg='red')
label_f.place(x=383, y=10)
filter_menu.place(x=360, y=70)
f_menu_btn = Button(app, text='Show', command=do_filter,fg='red')
f_menu_btn.place(x=380, y=120)

############################ histogram equalization
def histogram_equ():
    if image_text.get() != '':
        image = cv2.imread(image_text.get(),0)
        hiseq = cv2.equalizeHist(image)
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        plt.plot(cdf_normalized, color='red')
        plt.hist(image.flatten(), 256, [0, 256], color='black')
        plt.xlim([0, 256])
        plt.legend(('Cdf', 'Histogram'), loc='upper left')
        fig3 = plt.gcf()
        plt.show()
        plt.draw()
        fig3.savefig('histgraphsrc.png', dpi=100)
        hist, bins = np.histogram(hiseq.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max() / cdf.max()
        plt.plot(cdf_normalized, color='red')
        plt.hist(hiseq.flatten(), 256, [0, 256], color='black')
        plt.xlim([0, 256])
        plt.legend(('Cdf', 'Histogram'), loc='upper left')
        fig3 = plt.gcf()
        plt.show()
        plt.draw()
        fig3.savefig('histgraphequ.png', dpi=100)
        imgh = cv2.imread(r'histgraphsrc.png')
        imgh2 = cv2.imread(r'histgraphequ.png')
        imgh= cv2.cvtColor(imgh, cv2.COLOR_BGR2GRAY)
        imgh2= cv2.cvtColor(imgh2, cv2.COLOR_BGR2GRAY)
        imgh = cv2.resize(imgh, (400, 400))
        image= cv2.resize(image, (400, 400))
        hiseq = cv2.resize(hiseq, (400, 400))
        imgh2 = cv2.resize(imgh2, (400, 400))
        horizontalStack =np.concatenate((image,imgh,hiseq,imgh2), axis=1)
        horizontalStack = cv2.resize(horizontalStack, (1600,400))
        cv2.imshow('Original Image - Orjinal Image Histogram - Equalized Histogram Image - Equalized Histogram', horizontalStack)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(hiseq)

label_h = Label(app, text="Histogram \nEqualization",font="Times, 11",fg='green')
label_h.place(x=500, y=10)
histogram_button = Button(app, text='Show', font="Times, 10", command=histogram_equ,fg='green')
histogram_button.place(x=515, y=70)

#################################   Geometrical transformations
def do_transform():
    if geovar.get() == 'Resizing':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            resized = cv2.resize(image,(1200, 600))
            image = cv2.resize(image, (600, 500))
            if morvar.get() == 'Morphologicals' or morvar.get() =='! NONE !':  
                if filvar.get()!='! NONE !' or filvar.get()!='Filtering':
                    in_filt(resized)
                if filvar.get()=='! NONE !' or filvar.get()=='Filtering':   
                    cv2.imshow('Original Image', image)
                    cv2.imshow("Resized Image-%dpx-%dpx"% (1200,600),resized)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    save_image(resized)
            elif morvar.get() != 'Morphologicals' or morvar.get() !='! NONE !':
                in_mor(resized)
            else:
                cv2.imshow('Original Image', image)
                cv2.imshow("Resized Image-%dpx-%dpx"% (1200,600),resized)
                cv2.waitKey(0)
                cv2.destroyAllWindows()        
                save_image(resized)

    if geovar.get() == 'Rotation':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            rotated = imutils.rotate(image, angle=90)
            image = cv2.resize(image, (600, 500))
            rotated = cv2.resize(rotated, (600, 500))
            if morvar.get() == 'Morphologicals' or morvar.get() =='! NONE !':  
                if filvar.get()!='! NONE !' or filvar.get()!='Filtering':
                    in_filt(rotated)
                if filvar.get()=='! NONE !' or filvar.get()=='Filtering':   
                    cv2.imshow('Original Image', image)
                    cv2.imshow("Rotated 90°", rotated)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    save_image(rotated)
            elif morvar.get() != 'Morphologicals' or morvar.get() !='! NONE !':
                in_mor(rotated)
            else:
                cv2.imshow('Original Image', image)
                cv2.imshow("Rotated 90°", rotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                save_image(rotated)
    if geovar.get() == 'Cropping':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            cropped = image[0:500, 340:600]
            if morvar.get() == 'Morphologicals' or morvar.get() =='! NONE !':  
                if filvar.get()!='! NONE !' or filvar.get()!='Filtering':
                    in_filt(cropped)
                if filvar.get()=='! NONE !' or filvar.get()=='Filtering':   
                    cv2.imshow('Original Image', image)
                    cv2.imshow("Cropped", cropped)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    save_image(cropped)
            elif morvar.get() != 'Morphologicals' or morvar.get() !='! NONE !':
                in_mor(cropped)
            else:
                cv2.imshow('Original Image', image)
                cv2.imshow("Cropped", cropped)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                save_image(cropped)
    if geovar.get() == 'Swirl':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            image = image.astype('uint8')
            swirledd = swirl(image, rotation=0, strength=20, radius=120)
            image = cv2.resize(image, (600, 500))
            swirledd = cv2.resize(swirledd, (600, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Swirled", swirledd)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(swirledd)
    if geovar.get() == 'Flip':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            yansi = cv2.flip(image, 1)
            image = cv2.resize(image, (600, 500))
            yansi = cv2.resize(yansi, (600, 500))
            if morvar.get() == 'Morphologicals' or morvar.get() =='! NONE !':  
                if filvar.get()!='! NONE !' or filvar.get()!='Filtering':
                    in_filt(yansi)
                if filvar.get()=='! NONE !' or filvar.get()=='Filtering':   
                    cv2.imshow('Original Image', image)
                    cv2.imshow("Flip", yansi)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    save_image(yansi)
            elif morvar.get() != 'Morphologicals' or morvar.get() !='! NONE !':
                in_mor(yansi)
            else:
                cv2.imshow('Original Image', image)
                cv2.imshow("Flip", yansi)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                save_image(yansi)

            
            
transforms = {'Resizing', 'Rotation', 'Cropping', 'Swirl', 'Flip'}
geovar = StringVar()
geovar.set('Transforms')
geomenu = OptionMenu(app, geovar, *transforms)
label_g = Label(app, text="Select a Geometrical \n  Transformation", font="Times, 11",fg='blue')
label_g.place(x=640, y=10)
geomenu.place(x=660, y=68)
geo_menu_btn = Button(app, text='Show', command=do_transform,fg='blue')
geo_menu_btn.place(x=690, y=115)


#######################################  morphological

def do_morpho():
    if morvar.get() == 'Opening':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            kernel = np.ones((5, 5), np.uint8)
            img_opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            image = cv2.resize(image, (700, 500))
            img_opening = cv2.resize(img_opening, (700, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Openingg", img_opening)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_opening)
    if morvar.get() == 'Closing':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            kernel = np.ones((5, 5), np.uint8)
            img_closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            image = cv2.resize(image, (700, 500))
            img_closing = cv2.resize(img_closing, (700, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Closingg", img_closing)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_closing)
    if morvar.get() == 'Gradient':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            kernel = np.ones((5, 5), np.uint8)
            img_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
            image = cv2.resize(image, (700, 500))
            img_gradient = cv2.resize(img_gradient, (700, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Gradientt", img_gradient)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_gradient)
    if morvar.get() == 'Tophat':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            kernel = np.ones((15, 15), np.uint8)
            img_tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            image = cv2.resize(image, (700, 500))
            img_tophat = cv2.resize(img_tophat, (700, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Top Hatt", img_tophat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_tophat)
    if morvar.get() == 'Blackhat':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            kernel = np.ones((10, 10), np.uint8)
            img_blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            image = cv2.resize(image, (700, 500))
            img_blackhat = cv2.resize(img_blackhat, (700, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Black Hatt", img_blackhat)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_blackhat)
    if morvar.get() == 'Erosion':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            kernel = np.ones((5, 5), np.uint8)
            img_erosion = cv2.erode(image, kernel, iterations=1)
            image = cv2.resize(image, (700, 500))
            img_erosion = cv2.resize(img_erosion, (700, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Erosionn", img_erosion)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            img_erosion = cv2.cvtColor(img_erosion, cv2.COLOR_BGR2GRAY)
            save_image(img_erosion)
    if morvar.get() == 'Dilation':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            kernel = np.ones((5, 5), np.uint8)
            img_dilation = cv2.dilate(image, kernel, iterations=1)
            image = cv2.resize(image, (700, 500))
            img_dilation = cv2.resize(img_dilation, (700, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Dilationn", img_dilation)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_dilation)

    if morvar.get() == 'Crossing':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            kernel = np.ones((5, 5), np.uint8)
            img_crossing = cv2.morphologyEx(image, cv2.MORPH_CROSS, kernel)
            image = cv2.resize(image, (700, 500))
            img_crossing = cv2.resize(img_crossing, (700, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Crossingg", img_crossing)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_crossing)
    if morvar.get() == 'Ellipse':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            kernel = np.ones((8, 8), np.uint8)
            img_ellipse = cv2.morphologyEx(image, cv2.MORPH_ELLIPSE, kernel)
            image = cv2.resize(image, (700, 500))
            img_ellipse = cv2.resize(img_ellipse, (700, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Ellipsee", img_ellipse)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_ellipse)
    if morvar.get() == 'Rectangular':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            kernel = np.ones((5, 5), np.uint8)
            img_rectangular = cv2.morphologyEx(image, cv2.MORPH_RECT, kernel)
            image = cv2.resize(image, (700, 500))
            img_rectangular = cv2.resize(img_rectangular, (700, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Rectangularr", img_rectangular)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(img_rectangular)            

morphologicals = {'Opening','Closing','Gradient','Tophat','Blackhat','Erosion','Dilation','Crossing','Ellipse','Rectangular','! NONE !'}            
morvar = StringVar()
morvar.set('Morphologicals')
mormenu = OptionMenu(app, morvar, *morphologicals)
label_m = Label(app, text="Select a Morphological \n Operation", font="Times, 11",fg='purple')
label_m.place(x=840, y=10)
mormenu.place(x=860, y=68)
mor_menu_btn = Button(app, text='Show', command=do_morpho,fg='purple')
mor_menu_btn.place(x=885, y=115)            

#################################  intensity
def do_inten():
    if invar.get() == 'Piecewise':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            image = cv2.resize(image, (700, 500))
            pixelVal_vec = np.vectorize(pixelval)
            if pie_input.get()=='' and pie_input2.get() == '':
                piece=pixelVal_vec(image, 2500,3, 1, 1)
            elif pie_input.get()=='' and pie_input2.get()!='':
                piece=pixelVal_vec(image, 2500,int(pie_input2.get()), 1, 1)
            elif pie_input2.get()=='' and pie_input.get()!='':
                piece=pixelVal_vec(image,int(pie_input.get()),3, 1, 1)
            else:
                piece=pixelVal_vec(image, int(pie_input.get()), int(pie_input2.get()), 1, 1)
            piece = cv2.resize(piece, (700, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Piecewise", piece)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(piece)
    if invar.get() == 'Gamma':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            if gama_input.get() == '':
                gamma_corrected = np.array(255 * (image / 255) ** 0.8, dtype='uint8')
            else:
                gamma_corrected = np.array(255*(image / 255) ** float(gama_input.get()), dtype = 'uint8')   
            image = cv2.resize(image, (700, 500))
            gamma_image = cv2.resize(gamma_corrected, (700, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Power-Law (Gamma) Transformation", gamma_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(gamma_image)
    if invar.get() == 'Log Transform':
        if image_text.get() != '':
            image = cv2.imread(image_text.get())
            c = 255/(np.log(1 + np.max(image))) 
            log_transformed = c * np.log(1 + image)
            log_transformed = np.array(log_transformed, dtype = np.uint8)
            image = cv2.resize(image, (700, 500))
            logt = cv2.resize(log_transformed, (700, 500))
            cv2.imshow('Original Image', image)
            cv2.imshow("Log Transformation", logt)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            save_image(logt)

def pixelval(pix, r1, s1, r2, s2):
    if 0 <= pix <= r1:
        return (s1 / r1) * pix
    elif r1 < pix <= r2:
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2

gama_input = StringVar()
gama_label = Label(app, text='Gamma Value:', font="Times, 11",fg='purple') 
gama_label.place(x=325, y=340)
gama_entry = Entry(app, textvariable=gama_input)
gama_entry.place(x=450, y=340)
etc_g_label=Label(app, text='(etc: 2.3)', font="Times, 11")
etc_g_label.place(x=620, y=340)
###
pie_input=StringVar()
pie_label = Label(app, text='Piecewise Value 1:', font="Times, 11",fg='blue') 
pie_label.place(x=295, y=380)
pie_entry = Entry(app, textvariable=pie_input)
pie_entry.place(x=450, y=380)
etc_p_label=Label(app, text='(etc: 2000)', font="Times, 11")
etc_p_label.place(x=620, y=375)

pie_input2=StringVar()
pie_label2 = Label(app, text='Piecewise Value 2:', font="Times, 11",fg='blue') 
pie_label2.place(x=295, y=420)
pie_entry2 = Entry(app, textvariable=pie_input2)
pie_entry2.place(x=450, y=420)
etc_p_label2=Label(app, text='(etc: 2)', font="Times, 11")
etc_p_label2.place(x=620, y=418)


intensities = {'Piecewise','Gamma','Log Transform'}            
invar = StringVar()
invar.set('Intensities')
inmenu = OptionMenu(app, invar, *intensities)
label_i = Label(app, text="Select a Intensity Transformation", font="Times, 11",fg='purple')
label_i.place(x=415, y=230)
inmenu.place(x=450, y=275)
in_menu_btn = Button(app, text='Show', command=do_inten,fg='purple')
in_menu_btn.place(x=610, y=275) 

##################################################  active contour
def example_contour():
    image= cv2.imread(r'ronaldo.jpg')
    image= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('ronaldo_gray.png', image) 
    
    def circle_points(resolution, center, radius):
        radians = np.linspace(0, 2*np.pi, resolution)
        c = center[1] + radius*np.cos(radians)
        r = center[0] + radius*np.sin(radians)
        return np.array([c, r]).T
    points = circle_points(200, [50, 85], 50)[:-1] #this parameters are changeable(coordinates and size)
    def image_show(image, nrows=1, ncols=1, cmap='gray'):
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 12))
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        return fig, ax
    fig, ax = image_show(image)
    ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('ronaldo_circle.png', dpi=100)  #saving head circle image
    
    image_gray = color.rgb2gray(image)
    snake = seg.active_contour(image_gray, points,alpha=1,beta=0.8)
    fig, ax = image_show(image)
    ax.plot(points[:, 0], points[:, 1], '--r', lw=3)
    ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
    fig2 = plt.gcf()
    plt.show()
    plt.draw()
    fig2.savefig('ronaldo_actived.png', dpi=100)
    
    gray= cv2.imread(r'ronaldo_gray.png')
    gray = cv2.resize(gray, (500,600))
    circle= cv2.imread(r'ronaldo_circle.png')
    circle = cv2.resize(circle, (500, 600))
    actived= cv2.imread(r'ronaldo_actived.png')
    actived = cv2.resize(actived, (500, 600))
    horizontalStack = np.concatenate((gray,circle,actived), axis=1)
    horizontalStack = cv2.resize(horizontalStack, (1500, 600))
    cv2.imshow("Active Contour Example",horizontalStack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

act_label = Label(app, text="Active Contour Example", font="Times, 11",fg='red')
act_label.place(x=20, y=435)            
act_btn = Button(app, text='Show Example', command=example_contour,fg='red')
act_btn.place(x=50, y=480)
act_remind=Label(app, text="This example's photo is \n default. Your image \nentry doesn't use for\nthis example.", font="Times, 9")
act_remind.place(x=20, y=540)
################   video processing

def video_edge():
    if video_text.get() != '':
        cap = cv2.VideoCapture(video_text.get())
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
        frame_width = int(cap.get(3)) 
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height) 
        result = cv2.VideoWriter('videoresult2.mp4',  
                         cv2.VideoWriter_fourcc(*'MP4V'), 
                         20.0, (640,480)) 
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Using the Canny filter with different parameters
                edges_high_thresh = cv2.Canny(gray, 50,70)
                result.write(edges_high_thresh)
                images = np.hstack((gray, edges_high_thresh)) 
                # Display the resulting frame
                cv2.imshow('Video Edge Detection -- for exit press Q', images)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else: 
                break
        cap.release()
        cv2.destroyAllWindows()
        
        
        result3 = tkinter.messagebox.askquestion('SAVE Video', 'Do you want to save the video ?')
        if result3 == 'yes':
            result.release()
            tkinter.messagebox.showinfo('SAVED', 'Saving was completed successfully')
        else:
            tkinter.messagebox.showinfo('NOT SAVED', 'You did not save the video \n     You can continue')
        
        
        
vd_label = Label(app, text="Video Edge Detection", font="Times, 11",fg='green') 
vd_label.place(x=20, y=285)            
vd_btn = Button(app, text='Show', command=video_edge,fg='green')
vd_btn.place(x=80, y=330)

###### my filter for social media
def myfilter():
    if image_text.get() != '':
        image = cv2.imread(image_text.get())
        gamma_corrected = np.array(255 * (image / 255) ** 0.5, dtype='uint8')
        usmask = unsharp_mask(gamma_corrected, radius=20, amount=1)
        usmask = cv2.resize(usmask, (700, 600))
        image = cv2.resize(image, (700, 600))
        cv2.imshow("Original Photo", image)
        cv2.imshow("My Filter", usmask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(usmask)
myf_label= Label(app, text="My Filter \nfor Social Media", font="Times, 11",fg='brown') 
myf_label.place(x=840, y=230)  
myf_btn = Button(app, text='Show', command=myfilter,fg='brown')
myf_btn.place(x=884, y=295)


####after geometrical transforms,  for second morphological process function

def in_mor(image):
    if morvar.get() == 'Opening':
        kernel = np.ones((5, 5), np.uint8)
        img_opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Opening", img_opening)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(img_opening)
    if morvar.get() == 'Closing':
        kernel = np.ones((5, 5), np.uint8)
        img_closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Closing", img_closing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(img_closing)
    if morvar.get() == 'Gradient':
        kernel = np.ones((5, 5), np.uint8)
        img_gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Gradientt", img_gradient)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(img_gradient)
    if morvar.get() == 'Tophat':
        kernel = np.ones((15, 15), np.uint8)
        img_tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Tophat", img_tophat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(img_tophat)
    if morvar.get() == 'Blackhat':
        kernel = np.ones((10, 10), np.uint8)
        img_blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Blackhat", img_blackhat)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(img_blackhat)
    if morvar.get() == 'Erosion':
        kernel = np.ones((5, 5), np.uint8)
        img_erosion = cv2.erode(image, kernel, iterations=1)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Erosion", img_erosion)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(img_erosion)
    if morvar.get() == 'Dilation':
        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(image, kernel, iterations=1)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Dilation", img_dilation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(img_dilation)
    if morvar.get() == 'Crossing':
        kernel = np.ones((5, 5), np.uint8)
        img_crossing = cv2.morphologyEx(image, cv2.MORPH_CROSS, kernel)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Crossing", img_crossing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(img_crossing)
    if morvar.get() == 'Ellipse':
        kernel = np.ones((8,8), np.uint8)
        img_ellipse = cv2.morphologyEx(image, cv2.MORPH_ELLIPSE, kernel)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Ellipse", img_ellipse)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(img_ellipse)
    if morvar.get() == 'Rectangular':
        kernel = np.ones((5,5), np.uint8)
        img_rectangular = cv2.morphologyEx(image, cv2.MORPH_RECT, kernel)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Rectangular", img_rectangular)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(img_rectangular)


##### after geometrical transforms, second filter function

def in_filt(image):
    if filvar.get() == 'prewitt':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prew=prewitt(image)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Prewitt",  prew)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(prew)
    if filvar.get() == 'threshold':
        thresh = cv2.threshold(image,140,255, cv2.THRESH_BINARY)[1]
        cv2.imshow('Current Image', image)
        cv2.imshow("However Threshold",  thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(thresh)
    if filvar.get() == 'gabor':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        filt_gabor, filt_imag = gabor(image, frequency=1.48)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Gabor",  filt_gabor)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(filt_gabor)
    if filvar.get() == 'hessian':
        depth = cv2.CV_16S
        kernel_size = 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hess=hessian(image, sigmas=range(2,8,50))
        cv2.imshow('Current Image', image)
        cv2.imshow("However Hessian",  hess)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(hess)
    if filvar.get() == 'meijering':
        mei=meijering(image ,sigmas=range(1, 4, 500))
        cv2.imshow('Current Image', image)
        cv2.imshow("However Meijering", mei)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(mei)
    if filvar.get() == 'sobel':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sobell=sobel(image)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Sobel", sobell)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(sobell)
    if filvar.get() == 'unsharp_mask':
        usmask = unsharp_mask(image, radius=20, amount=1)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Unsharp_mask", usmask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(usmask)
    if filvar.get() == 'sato':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        satoo=sato(image, sigmas=range(5,11,4), black_ridges=True)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Satoo", satoo)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(satoo)
    if filvar.get() == 'median':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        med = median(image, disk(2.3))
        cv2.imshow('Current Image', image)
        cv2.imshow("However Median", med)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(med)
    if filvar.get() == 'roberts':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        robert=roberts(image, mask=None)
        cv2.imshow('Current Image', image)
        cv2.imshow("However Robert", robert)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        save_image(robert)



app.mainloop()