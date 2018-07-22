""" Prompt a window for free drawing and save drawing into Image"""
# Python 2
from Tkinter import *
import io
from PIL import Image
import numpy as np
from scipy.misc import imresize
from scipy.misc import imsave
from scipy.misc import imread
from scipy.misc import imshow
import pickle as pickle


class Drawing(object):
    DEFAULT_PEN_SIZE = 15.0
    DEFAULT_COLOR = 'black'
    LEARNING_RATE = 0.01

    def __init__(self, net):
        self.network = net
        self.root = Tk()

        self.btn_clear = Button(
            self.root, text='Clear Drawing', command=self.clear)
        self.btn_clear.grid(row=0, column=0)

        self.btn_save = Button(
            self.root, text='Save Drawing', command=self.save)
        self.btn_save.grid(row=0, column=1)

        self.btn_saveNetwork = Button(
            self.root, text='Save Network', command=self.save_network)
        self.btn_saveNetwork.grid(row=0, column=2)

        self.btn_validate = Button(
            self.root, text='Validate', command=self.validate)
        self.btn_validate.grid(row=1, column=0)

        self.label_result = Label(self.root, text="")
        self.label_result.grid(row=1, column=1)

        self.label_expect = Label(self.root, text="Expected Result : ")
        self.label_expect.grid(row=3, column=0)
        self.input_expect = Entry(self.root, width=10)
        self.input_expect.grid(row=3, column=1)
        self.btn_train = Button(self.root, text='Train', command=self.train)
        self.btn_train.grid(row=3, column=2)

        self.c = Canvas(self.root, bg='white', width=280, height=280)
        self.c.grid(row=4, columnspan=10)
        self.setup()

        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.DEFAULT_PEN_SIZE
        self.color = self.DEFAULT_COLOR
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def paint(self, event):
        self.line_width = self.DEFAULT_PEN_SIZE
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=self.DEFAULT_COLOR,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None

    def clear(self):
        self.c.delete("all")
        self.label_result['text'] = ''

    def save(self):
        # output image size
        size = 28, 28
        ps = self.c.postscript(colormode='gray')
        img = Image.open(io.BytesIO(ps.encode('utf-8'))).convert('L')
        img.thumbnail(size)
        img.save('my_drawing.jpg')

    def save_network(self):
        with open("my_network", 'wb') as f:
            pickle.dump(self.network, f)

    def validate(self):
        data = self.getInputValue()
        data = data.reshape((784, 1))
        result = self.network.guess(data)
        self.label_result['text'] = 'Result : {0}'.format(result)

    def train(self):
        input = int(self.input_expect.get())
        expected = self.network.vectorized_result(input)
        data = self.getInputValue()
        data = data.reshape((784, 1))
        self.network.update_mini_batch([[data, expected]], self.LEARNING_RATE)

    def getInputValue(self):
        """resize to 28x28, change to mono color, inverse color to white text"""
        size = 28, 28
        ps = self.c.postscript(colormode='mono')
        img = Image.open(io.BytesIO(ps.encode('utf-8'))).convert('L')
        img.thumbnail(size)
        data = np.array(img)
        data = self.inverse_color(data)
        return data

    def inverse_color(self, data):
        return 255-data
