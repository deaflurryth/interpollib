import matplotlib.pyplot as plt
from matplotlib import pyplot as mp
import numpy as np
import math
from math import sqrt, log
import csv 
import random
import pandas as pd
from pylab import mpl
from pandas import read_csv 
from scipy.optimize import curve_fit
#from tabulate import tabulate #pip install tabulate !!!
#from pylab import *
from typing import Tuple, List
import bisect
from os import path

def inter():
    print('Желательно использовать: %pylab и %matplotlib inline')
    np.seterr(divide= 'ignore', invalid= 'ignore')
    chose= 1
    if chose== 1:

        input1= int(input("Введите количество строк(желатеельно >50): "))
        main_matrix= []
        double1, double2= map(float, input("Введите интервал(2 числа через пробел): ").split())
        print('Выберите действие: ')
        print('[1] - Аппроксимация экспоненциальной функцикй | [2] - Аппроксимация линейной функцией')
        print('[3] - Интерполяция: NEWTONS METHOD            | [4] - Интерполяция: LAGRANGE METHOD')
        print('[5] - Функцией нормального распределения      | [6] - Аппроксимация логарифмической функцией')
        print('[7] - Аппроксимация тригонометрической ф-цей  | [8] - Метод Кубического сплайна')
        predator= int(input('=> '))
    #для ридера нам нужно что-то считывать, но считывать нам пока что нечего. 
    #именно поэтому создаем объект-вритер

        with open('m_matrix.csv', "w") as f: #без разделителя, обращаю вниманиие (no delimiter='' today)
            writer= csv.writer(f)

            y= []
            x= [_ for _ in range(0, input1)]

            for _ in range(input1):
                draft= random.uniform(double1, double2)
                draft= round(draft, 4)
                y.append(draft)

            ziper= zip(x,y) #зипуем
            for row in ziper:
                writer.writerow(row) 

        #а вот и обещанный ридер

        with open('m_matrix.csv', newline='') as File:  #на всякий случай: 
            reader= csv.reader(File)                    #newline='' служит убийцей проблемы с новой строкой
            for row in reader:
                #print(row)
                main_matrix.append(row)
        #print('_________________')
        #print('Исходная матрица: ')
        #for w in range(len(main_matrix)):
            #for z in range(len(main_matrix[w])):
                #print(main_matrix[w][z],end= ' ')
            #print() #принт для разделителя, без него сливаются воедино(почти)

        # -//-
        tablet= read_csv('m_matrix.csv')
        tablet.columns= ['X', 'Y']
        tablet.to_csv('m_matrix.csv')
        read_csv('m_matrix.csv')
        # -//-
        #print('_________________')
        #print('Выходные данные: ')
        x= np.array(x)
        y= np.array(y)
        #print(x)
        #print(y)
    elif chose== 2:
        uploader= input('Введите путь к файлу: /')
        if path.exists(uploader):
            print('Загрузка удалась')
        else:
            print('Вы ввели неверный путь, либо файла не существует')

        x= []
        y= []

        with open(uploader) as csvfile:
            reader= csv.reader(csvfile, quoting= csv.QUOTE_NONNUMERIC)
            #quoting=csv.QUOTE_NONNUMERIC здесь необходим на случай, если в вашем файле будут спец.
            #символы или прочие приколы
            for row in reader:
                for _ in row:
                    x.append(row[0])
                    y.append(row[1])

        del x[::2]
        del y[::2]
        #print(x)
        #print(y)
        input1= len(x)
    print('___________________')
    if predator== 1:
        def func(x, a, b, c, d):
            return a*np.exp(-c*(x*b))+d #формула
        popt, pcov= curve_fit(func, x, y, [100,400,0.001,0])
        def Gexp(popt,pcov, x):
            plt.title("Экспоненциальная аппроксимация", fontsize= 20, color= "orange")
            print(popt, plot(x,y, color= "black"))
            x= linspace(0, 1, input1)
            plt.plot(x, func(x, *popt), color= "r")
            plt.show() 
        Gexp(popt,pcov,x)

        sled= len(y)
        attack= list(range(1, sled+ 1))
        rows= zip(attack, x, y, func(x,*popt))
        with open('m_matrix.csv', mode = "w") as w_file:
            tabletka= csv.writer(w_file, lineterminator= "\r")
            tabletka.writerow(["Index", "x", "y",'y_vals'])
            for row in rows:
                tabletka.writerow(row)
        pd.read_csv('m_matrix.csv')
    elif predator== 2:
        def polynomial_fitting(data_x,data_y):
            #headpack
            size= len(data_x)     #размерность(то что мы подаем в input1 - кол-во строк)
            counter= 0            #счетчик
            sum_x= 0              #кол-во по x(опять же по input1)
            sum_sqare_x= 0        #кол-во по x^2
            sum_third_power_x= 0  #кол-во по x^3
            sum_four_power_x= 0   #кол-во по x^4
            average_x= 0          #среднее значение(нецелочисленное деление) по x
            average_y= 0          #среднее значение(нецелочисленное деление) по y
            sum_y= 0              #кол-во по y
            sum_xy= 0             #кол-во по x[n] * y[n]
            sum_sqare_xy= 0       #кол-во по (x[n]^2) * y[n]


            while counter< size: #пока каунтер не досчитает до длины строки(input1), цикл будет трудиться
                sum_x+= data_x[counter] 
                sum_y+= data_y[counter]
                sum_sqare_x+= math.pow(data_x[counter], 2)
                sum_third_power_x+= math.pow(data_x[counter],3)
                sum_four_power_x+= math.pow(data_x[counter],4)
                sum_xy+=data_x[counter]* data_y[counter]
                sum_sqare_xy+=math.pow(data_x[counter], 2)* data_y[counter]
                counter+= 1;

            average_x= sum_x/ size
            average_y= sum_y/ size

            return [[size, sum_x, sum_sqare_x, sum_y], 
                    [sum_x, sum_sqare_x, sum_third_power_x, sum_xy], 
                    [sum_sqare_x,sum_third_power_x,sum_four_power_x,sum_sqare_xy]]


        def calculate_parameter(data):

            i= 0;                 #отправная точка 1
            j= 0;                 #отправная точка 2 
            line_size= len(data)  #длина строки, в первой ячейке мы задали ее как input1

            while j< line_size-1: #потрясающий вложенный цикл, который функционирует на счетчиках
                line= data[j]
                temp= line[j]
                templete= []        #типа temp
                for x in line: #здесь инструкция такова, что мы попросту идем по лайну, который был
                               #задан предварительно. в даном случае - лайн задан датой, а дата задана
                               #предидущей функцией полиномиальной сборки
                    x= x/ temp
                    templete.append(x)
                data[j]= templete          
                flag= j+ 1    #логически следует, что это доп счетчик. для следующейго вложенного цикла

                while flag< line_size:
                    templete1= []        #temp
                    temp1= data[flag][j] # !!! очередной временный список из элементов счетчиков
                    i= 0   #каунтер
                    for x1 in data[flag]:
                        if x1!= 0:  #вот здесь реализована как раз таки сама соль лиинейной аппроксимации
                           x1= x1- (temp1* templete[i])
                                    #если вкратце то тут происходит интерполяция
                           templete1.append(x1)
                        else:
                           templete1.append(0)
                        i+= 1   #вот честно, мое хобби называть все счетчики каунтерами
                    data[flag]= templete1
                    flag+= 1 #каунтер
                j+= 1 #каунтер


           # -//-      

            parameters= [] #параметры.
            i= line_size- 1 #размерность, но на -1

            flag_j= 0 #этот флаг сослужит нам дело в будущем
            rol_size= len(data[0])
            flag_rol= rol_size- 2

            # -//-      

            while i>= 0:
                operate_line= data[i]
                if i== line_size-1:
                    parameter= operate_line[rol_size- 1]/ operate_line[flag_rol]
                    parameters.append(parameter) #заполенение пропущенных или непропущенных
                else:
                    flag_j=(rol_size-flag_rol-2) #вот тут и сослужил
                    temp2= operate_line[rol_size-1]

                    result_flag= 0 #total
                    while flag_j> 0: #я начинаю путаться в математических терминах...
                        temp2-= operate_line[flag_rol+ flag_j]* parameters[result_flag]
                        result_flag+= 1
                        flag_j-= 1
                    parameter= temp2/ operate_line[flag_rol]
                    parameters.append(parameter)
                flag_rol-= 1
                i-= 1 
            return parameters


        def calculate(data_x,parameters):
            datay= []
            for x in data_x:
                datay.append(parameters[2]+ parameters[1]* x+ parameters[0]* x* x)
            return datay




        def Glin(data_x, data_y_new, data_y_old):
            plt.plot (data_x, data_y_new, label= "Кривая подгонки", color= "black")
            plt.scatter (data_x, data_y_old, label= "Данные", color= "r")
            #mpl.rcParams['axes.unicode_minus']= False
            plt.title("Данные подгонки полинома с одной переменной", fontsize= 20, color= "orange")
            plt.legend(loc= "upper left")
            #plt.show()

        # -//-
        data= polynomial_fitting(x, y)
        parameters= calculate_parameter(data)
        # -//-

        for _ in parameters:
            print(_)
        print(parameters)

        # -//-
        newData= calculate(x,parameters)
        Glin(x,newData,y)
    elif predator== 3:
        def differ(x, y): #вычисление коэффициента разности
            counter= 0 #каунтер
            ratio= input1* [0]
            while counter< input1- 1:
                decount= input1- 1
                while decount> counter:
                    if counter== 0:
                        ratio[decount]= ((y[decount]- y[decount-1])/ (x[decount]- x[decount- 1])) 
                    else:
                        ratio[decount]= (ratio[decount]- ratio[decount- 1])/ (x[decount]-x[decount- 1- counter])
                    decount-= 1
                counter+= 1
            return ratio

        def function(data):

            return x[0]+ parameters[1]* (data-0.4)+ parameters[2]* (data-0.4)* (data-0.55)+\
                   parameters[3]* (data-0.4)* (data-0.55)* (data-0.65)\
                   +parameters[4]* (data-0.4)* (data-0.55)* (data-0.80)


        def d_rejoin(x, parameters): #благодаря этой функции перекидываем все выше получвшеееся
            rejoinData= []
            for data in x:
                rejoinData.append(function(data))
            return rejoinData


        def Gnew(newData): #построение плота
            plt.scatter(x, y, label= "Данные", color= "r")
            plt.plot(x, newData, label= "Кривая подгонки", color= "black")
            plt.scatter(0,1, label= "Точка функции предск.", color= "g")
            plt.title("Метод интерполяции Ньютона", fontsize= 20, color= "orange")
            #mpl.rcParams['axes.unicode_minus']= False
            plt.legend(loc= "upper left")
            #plt.show()


        parameters= differ(x, y)

        hatedmethod= d_rejoin(x, parameters)
        Gnew(hatedmethod)
    elif predator== 4:
        import matplotlib.pyplot as plt

        def lagran(x,y,t):
            z=0
            for j in range(len(y)):
                p1=1; p2=1
                for i in range(len(x)):
                    if i==j:
                        p1=p1*1; p2=p2*1   
                    else: 
                        p1=p1*(t-x[i])
                        p2=p2*(x[j]-x[i])
                z=z+y[j]*p1/p2
            return z
        xnew=np.linspace(np.min(x),np.max(x))
        ynew=[lagran(x,y,i) for i in xnew]
        plt.title("Метод Лагранжа", fontsize= 20, color= "orange")
        plt.plot(x,y,'o',xnew,ynew, color= 'r', mec= 'blue')
        plt.grid()
        plt.show()
    elif predator== 5:
        import matplotlib.pyplot as plt
        def Gauss(x, x0,sigma):
            return np.exp(-np.power((x- x0)/ sigma, 2.)/ 2.)

        x_values= np.linspace(-3, 3, 120)
        space= [[w,z] for w, z in zip(x, y)]
        for my, sig in space:
            mp.plot(x_values, Gauss(x_values, my, sig))

        plt.title("ф. Нормального распределения", fontsize= 20, color= "orange")
        plt.show()
        tetra= Gauss(x,y,sig)
    elif predator== 6:
        import matplotlib.pyplot as plt
        def applog(n, x1):
            a= (1+ x1)/ 2
            b= sqrt(x1)
            for i in range(n):
                a= (a+ b)/ 2
                b= sqrt(a* b)
            return (x1 - 1)/ a

        allowed_n= [_ for _ in range(2, input1)]
        colors= 'rgby'* input1
        x1= np.linspace(10, 500, 100)

        def Glog(allowed_n, colors):
            fig= plt.figure()
            ax= fig.add_subplot()
            x1= np.linspace(10, 500, 100)
            for n, c in zip(allowed_n, colors):
                ax.plot(x1, applog(n, x1), color= c, label= n)
            #plt.legend(title='L', ncol= input1//4, facecolor= 'none')
            #plt.show() 

            #fig.set_figwidth(50)
            #fig.set_figheight(50)
        Glog(allowed_n, colors)
    elif predator== 7:
        import matplotlib.pyplot as plt
        def sigmoid(x): #монотонно возрастающая
            return 1.0 / (1.0 + np.exp(-x)) 


        def sigmoid_prime(x):
            return sigmoid(x) * (1 - sigmoid(x))


        def linear(x):
            return x

        def linear_prime(x):
            return 1

        def tanh(x):
            return (np.exp(x)- np.exp(-x))/(np.exp(x) + np.exp(-x))

        def tanh_prime(x):
            return 1- tanh(x)* tanh(x)

        class Network: 
            def __init__(self, sizes, activation_func = sigmoid, activation_prime = sigmoid_prime):

                self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
                self.weights = [np.random.randn(y, x) for x, y in zip(sizes, sizes[1:])]

                self.num_layers = len(sizes)
                self.sizes = sizes

                self.activation_function = activation_func
                self.actiovation_prime = activation_prime

            def forward_prop(self, a):
                for w, b in zip(self.weights, self.biases):
                    a = self.activation_function(np.dot(w, a) + b)
                return a

            def cost_derivative(self, output_activations, y):
                return (output_activations - y)

            def backprop(self, x, y): #функция для единичного примера

                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]

             #перекдиываем вперед
                activation = x  #это типа пусковой крючек
                a_mas = [x]
                z_mas = []

                for b, w in zip(self.biases, self.weights):
                    z = np.dot(w, activation) + b
                    activation = self.activation_function(z)
                    z_mas.append(z)
                    a_mas.append(activation)
                    pass

                #пасс назад
                delta = self.cost_derivative(a_mas[-1], y) * self.actiovation_prime(z_mas[-1])
                nabla_b[-1] = delta
                nabla_w[-1] = np.dot(delta, a_mas[-2].T)

                for l in range(2, self.num_layers):
                    delta = np.dot(self.weights[-l + 1].transpose(), delta) * self.actiovation_prime(z_mas[-l])
                    nabla_b[-l] = delta
                    nabla_w[-l] = np.dot(delta, a_mas[-l - 1].T)

                return nabla_b, nabla_w
            def update_mini_batch(self, mini_batch, eta):

                nabla_b= [np.zeros(b.shape) for b in self.biases]
                nabla_w= [np.zeros(w.shape) for w in self.weights]

                for x, y in mini_batch:
                    delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                    nabla_b= [nb+ dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w= [nw+ dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

                eps = eta / len(mini_batch)
                self.weights= [w- eps* nw for w, nw in zip(self.weights, nabla_w)]
                self.biases= [b- eps* nb for b, nb in zip(self.biases, nabla_b)]

            def SGD(self, training_data, epochs, mini_batch_size, eta):
                n= len(training_data)

                for j in range(epochs):
                    random.shuffle(training_data)
                    mini_batches= [training_data[k:k+ mini_batch_size]
                                    for k in range(0, n, mini_batch_size)]

                    for mini_batch in mini_batches:
                        self.update_mini_batch(mini_batch, eta)


        net2= Network([1,input1,1])
        #x = np.linspace(0,10,1000)
        y2= np.sin(x)
        train= [(np.array(x[i]).reshape(1,1),np.array(y2[i]).reshape(1,1)) for i in range(len(x))]       
        net2.SGD(train, input1, 10, 0.1)

        y_pred= []
        y_tmp= []

        for _ in range(len(x)):
            y_tmp.append(net2.forward_prop(train[_][0]))
            y_pred.append(float(net2.forward_prop(train[_][0])))

        def Gtrin(x, y, y_pred):
            plt.title("Тригонометрия", fontsize= 20, color= "orange")
            plt.plot(x, y, 'r', x, y_pred, mec= 'blue', ms= 4)
            #plt.show()
        Gtrin(x,y, y_pred)
        def Gtrin2(x, y, y_pred):
            plt.title("Тригонометрия", fontsize= 20, color= "orange")
            ax.plot(x, y, 'r', x, y_pred, mec= 'blue', ms= 4)
            #plt.show()
        #Gtrin2(x,y, y_pred)
    elif predator== 8:
        import matplotlib.pyplot as plt
        def compute_changes(x: List[float]) -> List[float]:
            return [x[i+1] - x[i] for i in range(len(x) - 1)]

        def create_tridiagonalmatrix(n: int, h: List[float]) -> Tuple[List[float], List[float], List[float]]:
            A = [h[i] / (h[i] + h[i + 1]) for i in range(n - 2)] + [0]
            B = [2] * n
            C = [0] + [h[i + 1] / (h[i] + h[i + 1]) for i in range(n - 2)]
            return A, B, C

        def create_target(n: int, h: List[float], y: List[float]):
            return [0] + [6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]) / (h[i] + h[i-1]) for i in range(1, n - 1)] + [0]

        def solve_tridiagonalsystem(A: List[float], B: List[float], C: List[float], D: List[float]):
            c_p = C + [0]
            d_p = [0] * len(B)
            X = [0] * len(B)

            c_p[0] = C[0] / B[0]
            d_p[0] = D[0] / B[0]
            for i in range(1, len(B)):
                c_p[i] = c_p[i] / (B[i] - c_p[i - 1] * A[i - 1])
                d_p[i] = (D[i] - d_p[i - 1] * A[i - 1]) / (B[i] - c_p[i - 1] * A[i - 1])

            X[-1] = d_p[-1]
            for i in range(len(B) - 2, -1, -1):
                X[i] = d_p[i] - c_p[i] * X[i + 1]

            return X

        def compute_spline(x: List[float], y: List[float]):
            n = len(x)
            if n < 3:
                raise ValueError('Too short an array')
            if n != len(y):
                raise ValueError('Array lengths are different')

            h = compute_changes(x)

            A, B, C = create_tridiagonalmatrix(n, h)
            D = create_target(n, h, y)

            input1 = solve_tridiagonalsystem(A, B, C, D)

            coefficients = [[(input1[i+1]-input1[i])*h[i]*h[i]/6, input1[i]*h[i]*h[i]/2, 
                             (y[i+1] - y[i] - (input1[i+1]+2*input1[i])*h[i]*h[i]/6), 
                             y[i]] for i in range(n-1)]

            def spline(val):
                idx = min(bisect.bisect(x, val)-1, n-2)
                z = (val - x[idx]) / h[idx]
                C = coefficients[idx]
                return (((C[0] * z) + C[1]) * z + C[2]) * z + C[3]

            return spline

        # -//-
        x = [i for i in range(0, input1)]
        spline = compute_spline(x, y)
        # -//-

        colors = 'r'
        for i, x in enumerate(x):
            assert abs(y[i] - spline(x)) < 1e-8, f'Error at {x}, {y[i]}'

        x_vals = [v / 10 for v in range(input1)]
        y_vals = [spline(y) for y in x_vals]
        def Gcub():
            x_vals = [v / 10 for v in range(input1)]
            y_vals = [spline(y) for y in x_vals]
            plt.scatter(x_vals, y_vals, c=colors)
            plt.title("Кубический сплайн", fontsize= 20, color= "orange")
            plt.plot(x_vals,y_vals, 'black')
            #print(x_vals,y_vals)
            #plt.show()
        Gcub()
        def Gcub2():
            x_vals = [v / 10 for v in range(input1)]
            y_vals = [spline(y) for y in x_vals]
            ax.scatter(x_vals, y_vals, c=colors)
            plt.title("Кубический сплайн", fontsize= 20, color= "orange")
            ax.plot(x_vals,y_vals, 'black')
            #print(x_vals,y_vals)
            #plt.show()
        #Gcub2()
#inter()