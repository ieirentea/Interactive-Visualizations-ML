#second and third visualizations are adapted from TI3145TU Machine Learning and Introduction to AI course from TU Delft
import numpy as np
import plotly.express as px
import ipywidgets as widgets
import plotly.graph_objects as go
from IPython.display import display, clear_output
import seaborn as sns
import sklearn
import os
np.random.seed(42)
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
import sklearn.metrics
import numpy as np
from ipywidgets import interactive
import matplotlib.pyplot as plt
import sklearn.metrics
from ipywidgets import interact_manual

points = None
X_b = None
X = None
y = None
X_new_b = None
X_new = None
pa = None
pb = None
zz_P = None
par_range = None
MSE_best = None
theta_list = None
MSE_list = None

# Define the cost functions & corresponding gradients
def lsm_function(x):
    return x**2

def lsm_gradient(x):
    return 2*x

def cubic_function(x):
    return x**3

def cubic_gradient(x):
    return 3*x**2

def sine_function(x):
    return np.sin(x)

def sine_gradient(x):
    return np.cos(x)

def custom_function(x):
    return x**4 - 4*x**2 + x

def custom_gradient(x):
    return 4*x**3 - 8*x + 1

def linear_function(x):
    return -2*x

def linear_gradient(x):
    return -2

def compute_next_iteration(current_point, learning_rate, gradient_function):
    new_point = current_point - learning_rate * gradient_function(current_point)
    return new_point

def draw_cost_function(cost_function, start_point, learning_rate, iterations, lines):
    function = {
        'Least Squares': lsm_function,
        'Cubic Function': cubic_function,
        'Sinus Function': sine_function,
        'Custom Function': custom_function,
        'Linear Function': linear_function
    }[cost_function]
    
    gradient = {
        'Least Squares': lsm_gradient,
        'Cubic Function': cubic_gradient,
        'Sinus Function': sine_gradient,
        'Custom Function': custom_gradient,
        'Linear Function': linear_gradient
    }[cost_function]
    
    points.clear()
    points.append(start_point)
    
    for i in range(iterations-1):
        new_point = compute_next_iteration(points[-1], learning_rate, gradient)
        points.append(new_point)
    
    x_values = np.linspace(-10, 10, 100)
    if points[-1]<-10:
        x_values = np.linspace(points[-1],10,100)
    elif points[-1]>10:
        x_values = np.linspace(-10,points[-1],100)
    
    if cost_function == 'Custom Function':
        x_values = np.linspace(-3, 3, 100)
    y_values = function(x_values)
    
    p = sns.lineplot(x = x_values, y = y_values)
    p.plot(points[:-1], function(np.array(points[:-1])), 'go')
    p.plot(points[-1], function(np.array(points[-1])), 'rD')
    p.set_xlabel('x')
    p.set_ylabel('loss')

    
    if lines and len(points)>1:
        point = points[-2]
        gradient_at_point = gradient(point)
        tangent_slope = gradient_at_point
        tangent_intercept = function(point) - tangent_slope * point
        tangent_line_x = np.linspace(point - 2, point + 2, 100)
        tangent_line_y = tangent_slope * tangent_line_x + tangent_intercept
        p.plot(tangent_line_x, tangent_line_y, color='g')
        
    if len(points)>1:
        display(f"Last update: new value({points[-1]:.2f}) = old value({points[-2]:.2f}) - learning rate({learning_rate:.2f}) * gradient({gradient(points[-2]):.2f})")
    

def GD_2D():
    start_point_slider = widgets.FloatSlider(value=2.0, min=-10.0, max=10.0, step=0.1, description='Starting Value:', 
                                             layout = widgets.Layout(width='600px'), style= {'description_width': 'initial'}, 
                                             continuous_update=False)
    learning_rate_slider = widgets.FloatSlider(value=0.1, min=0.01, max=1.0, step=0.01, description='Learning Rate:', 
                                               layout = widgets.Layout(width='600px'), style= {'description_width': 'initial'},
                                               continuous_update=False)
    cost_function_dropdown = widgets.Dropdown(options=[
        'Least Squares',
        'Cubic Function',
        'Sinus Function',
        'Custom Function',
        'Linear Function']
    , value = 'Least Squares', description='Selected Cost Function:', style= {'description_width': 'initial'})
    iterations_slider = widgets.IntText(value = 1, min = 1, max = 30, step = 1, description = 'Number of iterations displayed: ', 
                                          layout = widgets.Layout(width='300px'), style= {'description_width': 'initial'}, 
                                          continuous_update=False)
    lines_checkbox = widgets.Checkbox(value=False, description='Display on plot the gradient used in the last update.', 
                                      style= {'description_width': 'initial'}, layout = widgets.Layout(width='350px'),
                                      disabled=False)

    global points
    points = []

    widget_box = widgets.VBox([start_point_slider, learning_rate_slider, cost_function_dropdown, iterations_slider, lines_checkbox])

    out = widgets.interactive_output(draw_cost_function, {'cost_function' : cost_function_dropdown, 'start_point':
                                                        start_point_slider, 'iterations': iterations_slider, 
                                                        'learning_rate' : learning_rate_slider, 'lines': lines_checkbox})
    return display(widget_box, out)


#code adapted from TI3145TU Machine Learning and Introduction to AI course from TU Delft
def GD_3D():
    
    global X_b
    global X
    global y
    global X_new_b
    global X_new
    global pa
    global pb
    global zz_P
    global par_range
    global MSE_best
    global theta_list
    global MSE_list

    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 to each instance
    theta_path_sgd = []
    m = len(X_b)
    np.random.seed(42)
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance

    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    X_new = np.array([[-2], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # add x0 = 1 to each instance

    par_range = [-10,10,-15,15] # b, a

    pb, pa = np.meshgrid(np.linspace(par_range[0], par_range[1], 200).reshape(-1, 1),
        np.linspace(par_range[2], par_range[3], 200).reshape(-1, 1))
    P = np.c_[pb.ravel(), pa.ravel()]

    Y_P = np.matmul(P,X_b.T)
    bla = Y_P-y.T
    SE_P = bla**2
    MSE_P = np.mean(SE_P,axis=1)

    zz_P = MSE_P.reshape(pb.shape)

    theta_best
    y_predict_best = X_b.dot(theta_best)
    MSE_best = sklearn.metrics.mean_squared_error(y, y_predict_best)
    
    wa = widgets.FloatSlider(value=-10,min=-15, max=15, step=0.1, continuous_update=False, 
                             style= {'description_width': 'initial'}, description='Starting value a')
    wb = widgets.FloatSlider(value=-5,min=-10, max=10, step=0.1, continuous_update=False,
                             style= {'description_width': 'initial'}, description='Starting value b')
    learning_rate = widgets.FloatLogSlider(value=0.01, base=10, min=-5, max=-0.1, step=0.2, 
                                           description='Learning rate', continuous_update=False)
    epochs = widgets.IntSlider(value=5, min=1, max=50, continuous_update=False, description='Epochs')

    init_box = widgets.HBox([wa, wb])
    hyp_box = widgets.HBox([learning_rate, epochs])
    ui = widgets.VBox([init_box, hyp_box])

    out = widgets.interactive_output(sgd, {'a':wa, 'b':wb, 'learning_rate': learning_rate, 'epochs': epochs})

    return display(ui, out)

def sgd(a, b, learning_rate, epochs):
    
    global theta_list
    global MSE_list
    theta_list = np.array([[b],[a]]).T

    theta = np.array([[b],[a]])
    y_pred = X_b.dot(theta)
    MSE = sklearn.metrics.mean_squared_error(y, y_pred)

    MSE_list = [MSE]
    
    n_epochs = epochs
    m = len(X_b)
    theta_path_sgd = []
    
    theta = np.array([[b],[a]])
    
    for epoch in range(n_epochs):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - learning_rate * gradients
        theta_path_sgd.append(theta)

        y_predict = X_b.dot(theta) 
        MSE = sklearn.metrics.mean_squared_error(y, y_predict)
        MSE_list.append(MSE)

        theta_list = np.concatenate((theta_list,theta.T),axis=0)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
    
    y_predict = X_new_b.dot(theta)           
    style = "r--"        
    ax1.plot(X_new, y_predict, style)   
    
    ax1.plot(X, y, "b.")                                 
    ax1.set_xlabel("$x$", fontsize=18)                     
    ax1.set_ylabel("$y$", rotation=0, fontsize=18)           
    ax1.axis([0, 2, 0, 15])    
    
    CS = ax2.contour(pb, pa, zz_P, levels=10)
    ax2.clabel(CS, inline=True, fontsize=10)
    ax2.scatter(theta_list[:,0],theta_list[:,1])
    ax2.scatter(theta_list[-1,0],theta_list[-1,1],c='r')
    ax2.axis(par_range)
    ax2.set_xlabel("$b$")
    ax2.set_ylabel("$a$", rotation=0)
    ax2.set_title('parameter space')
    
    ax3.set_title('Training Curve, MSE after training: %.1f' % MSE)
    ax3.plot(MSE_list,'o-')
    x_best_plot = np.array([0, len(MSE_list)])
    y_best_plot = np.array([MSE_best,MSE_best])
    ax3.plot(x_best_plot,y_best_plot)
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('MSE')
    
def GD_manual():

    global X_b
    global X
    global y
    global X_new_b
    global X_new
    global pa
    global pb
    global zz_P
    global par_range
    global MSE_best
    global theta_list
    global MSE_list
    global w1_label
    global w2_label
    global w3_label
    global w4_label
    global w5_label
    
    global wb_label

    X = 2 * np.random.rand(100, 1) - 1

    feat_extract = sklearn.preprocessing.PolynomialFeatures(degree=5,include_bias=False)
    X_2 = feat_extract.fit_transform(X)
    X_b = np.c_[np.ones((100, 1)), X_2]  # add x0 = 1 to each instance

    theta_opt = np.array([10,1,5,8,6,8])
    y = np.matmul(X_b,theta_opt) + 0.25*np.random.randn(100, 1).T
    y = y.T

    par_range = [-10,10,-15,15]
    
    theta_best = theta_opt
    theta_init = theta_best.copy()*0

    theta_list = theta_init.copy()

    theta = theta_init.copy()
    y_trn = X_b.dot(theta)
    MSE = sklearn.metrics.mean_squared_error(y, y_trn)

    MSE_list = [MSE]

    y_predict_best = X_b.dot(theta_best)
    MSE_best = sklearn.metrics.mean_squared_error(y, y_predict_best)

    w1 = widgets.FloatSlider(min=par_range[2], max=par_range[3], step=0.1, continuous_update=False, value=theta_init[0])
    w2 = widgets.FloatSlider(min=par_range[2], max=par_range[3], step=0.1, continuous_update=False, value=theta_init[1])
    w3 = widgets.FloatSlider(min=par_range[2], max=par_range[3], step=0.1, continuous_update=False, value=theta_init[2])
    w4 = widgets.FloatSlider(min=par_range[2], max=par_range[3], step=0.1, continuous_update=False, value=theta_init[3])
    w5 = widgets.FloatSlider(min=par_range[2], max=par_range[3], step=0.1, continuous_update=False, value=theta_init[4])

    wb = widgets.FloatSlider(min=par_range[0], max=par_range[1], step=0.1, continuous_update=False, value=theta_init[5])
    w1_label = widgets.Label(value="w1")
    w2_label = widgets.Label(value="w2")
    w3_label = widgets.Label(value="w3")
    w4_label = widgets.Label(value="w4")
    w5_label = widgets.Label(value="w5")

    wb_label = widgets.Label(value="w0")

    bar_1 = widgets.HBox([w1,w1_label])
    bar_2 = widgets.HBox([w2,w2_label])
    bar_3 = widgets.HBox([w3,w3_label])
    bar_4 = widgets.HBox([w4,w4_label])
    bar_5 = widgets.HBox([w5,w5_label])

    bar_b = widgets.HBox([wb,wb_label])

    X_new = np.arange(-1,1,0.01).reshape(-1,1)
    X_new_2 = feat_extract.transform(X_new)
    X_new_b = np.c_[np.ones((len(X_new_2), 1)), X_new_2]  # add x0 = 1 to each instance
    
    ui = widgets.VBox([bar_b, bar_1, bar_2, bar_3, bar_4, bar_5])

    out = widgets.interactive_output(h, {'w1':w1, 'w2':w2, 'w3': w3, 'w4': w4, 'w5': w5, 'wb': wb})
    return display(ui, out)

def h(w1, w2, w3, w4, w5, wb):

    global theta_list  
    global MSE_list
    theta = np.array([[wb],[w1],[w2],[w3],[w4],[w5]])
    
    y_trn = np.matmul(X_b,theta)
    MSE = sklearn.metrics.mean_squared_error(y, y_trn)
    MSE_list.append(MSE)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    
    y_predict = np.matmul(X_new_b,theta)
    
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)
    grad_norm = np.sqrt(np.sum(gradients**2))
    gradients_unit = 1/grad_norm*gradients.copy()
    
    global w1_label
    global w2_label
    global w3_label
    global w4_label
    global w5_label
    
    global wb_label
    
    wb_label.value = ('-dJ/dw0 = %+f' % -gradients[0,0])
    w1_label.value = ('-dJ/dw1 = %+f' % -gradients[1,0])
    w2_label.value = ('-dJ/dw2 = %+f' % -gradients[2,0])
    w3_label.value = ('-dJ/dw3 = %+f' % -gradients[3,0])
    w4_label.value = ('-dJ/dw4 = %+f' % -gradients[4,0])
    w5_label.value = ('-dJ/dw5 = %+f' % -gradients[5,0])
    

    ax1.plot(X_new, y_predict, "r-", linewidth=2, label="Predictions")
    ax1.plot(X, y, "b.")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$", rotation=0)
    ax1.legend(loc="upper left", fontsize=14)
    ax1.axis([-1, 1, 0, 40])
    ax1.set_title('regression problem')
    
    ax2.set_title('Training Curve, current MSE: %.1f' % MSE)
    ax2.plot(MSE_list)
    x_best_plot = np.array([0, len(MSE_list)])
    y_best_plot = np.array([MSE_best,MSE_best])
    ax2.plot(x_best_plot,y_best_plot)
    ax2.set_xlabel('tries')
    ax2.set_ylabel('MSE')