import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import ipywidgets as widgets

global errors
global slopes
global variances

def read_data(file_name):
    """
    This function loads a given matrix data file into a numpy matrix.
    :param file_name: name of the file to be read
    :return: the data as a numpy array
    """
    lines = [line.rstrip('\n') for line in open(file_name)]

    result = np.zeros((len(lines), len(lines[0].split(" "))))

    for (i, line) in enumerate(lines):
        line = line.split(" ")
        for (j, number) in enumerate(line):
            result[i][j] = float(number)

    return result

def project_point(slope, p):
    A = np.array((1, slope))
    v = np.dot(1/np.sqrt(1+slope**2),A)
    return np.dot(np.dot(p, v),v)
    

def draw_line(slope, intercept=0):
    global errors, slopes, variances
    #plot data points
    data = read_data("gaussian.txt")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
    x = [[el[0]] for el in data]
    y = [[el[1]] for el in data]
    x = x - np.mean(x)
    y = y - np.mean(y)
    ax1.scatter(x, y, c = 'dodgerblue', marker = 'o', alpha = 0.4)
    
    #plot line with given slope and intercept 0
    x_vals = np.array(ax1.get_xlim())
    y_vals = intercept + slope * x_vals
    
    p1 = np.array(x_vals[0], y_vals[0])
    p2 = np.array(x_vals[1], y_vals[1])
    
    error = 0
    errors = []
    variances = []
    
    #generate line
    slopes = np.linspace(-2, 2, 50)
    for i in slopes:
        error = 0
        projected_x = []
        projected_y = []
        projected_points = []
        for point in data:
            projected_points.append(project_point(i, point))
            projected_x.append(projected_points[-1][0])
            projected_y.append(project_point(i, point)[1])
            distance = ((point[0] - projected_x[-1])**2 + (point[1] - projected_y[-1])**2)**0.5
            error += distance

        errors.append(error)
        variance = np.var(projected_points)
        variances.append(variance)
    
    #generate current point
    projected_x = []
    projected_y = []
    projected_points = []
    error = 0
    for point in data:
        projected_points.append(project_point(slope, point))
        projected_x.append(projected_points[-1][0])
        projected_y.append(project_point(slope, point)[1])
        distance = ((point[0] - projected_x[-1])**2 + (point[1] - projected_y[-1])**2)**0.5
        error += distance
    
    ax1.scatter(projected_x, projected_y, c = 'dodgerblue', marker = 'o')
    
    x_vals = np.array(ax1.get_xlim())
    y_vals = intercept + slope * x_vals
    ax1.plot(x_vals, y_vals, '--', color = 'coral')
    
    variance = np.var(projected_points)
    
    np.var(projected_x)
    
    ax2.plot(slopes, errors, c = 'grey') #lightsalmon
    ax2.scatter(slope, error, c = 'orangered', marker = 'o')
    ax2.title.set_text('Reconstruction error')
    ax2.set_xlabel("slope")
    ax2.set_ylabel("error")
    ax3.plot(slopes, variances, c = 'grey') #yellowgreen
    ax3.scatter(slope, variance, c = 'olivedrab', marker = 'o')
    ax3.title.set_text('Variance')
    ax3.set_xlabel("slope")
    ax3.set_ylabel("variance")

def pca_vis1():
    errors =  []
    slopes = []
    variances = []

    gradient_slider = widgets.FloatSlider(
        value=0.5,
        min=-2.0,
        max=2.0,
        step=0.1,
        description='Line gradient:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout = widgets.Layout(width='800px')
    )
    out = widgets.interactive_output(draw_line, {'slope' : gradient_slider})
    return display(gradient_slider, out)


def draw_scale(scalex, scaley):
    data = read_data("gaussian.txt")
    x = [[el[0]] for el in data]
    y = [[el[1]] for el in data]
    x = np.array(x)
    y = np.array(y)
    x = x * scalex
    y = y * scaley
    x = x - np.mean(x)
    y = y - np.mean(y)
    
    pca =PCA()
    pca.fit(np.column_stack((x, y)))
    components = pca.components_
    plt.scatter(x, y, c = 'dodgerblue', marker = 'o', alpha = 0.4)
    
    #draw pc1
    x1=0
    x2=components[0,0]
    y1=0
    y2=components[0,1]
    line_eqn = lambda x : ((y2-y1)/(x2-x1)) * (x - x1) + y1        
    xrange = np.arange(x.min(),x.max(),0.2)
    
    plt.plot(xrange, [ line_eqn(x) for x in xrange], color='r', linestyle='-', linewidth=2, label = 'PC1')
    
    #draw pc2
    x1=0
    x2=components[1,0]
    y1=0
    y2=components[1,1]
    line_eqn = lambda x : ((y2-y1)/(x2-x1)) * (x - x1) + y1        
    xrange = np.arange(y.min(),y.max(),0.2)
    
    plt.plot(xrange, [ line_eqn(x) for x in xrange], color='g', linestyle='--', linewidth=2, label = 'PC2')
    plt.title("Plot with x scaled by " + str(scalex) + " and y scaled by " + str(scaley))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

def pca_scaling():
    scale_x_slider = widgets.FloatSlider(
        value=1,
        min=0.0,
        max=5.0,
        step=0.2,
        description='Scale x with:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout = widgets.Layout(width='510px')
    )
    scale_y_slider = widgets.FloatSlider(
        value=1,
        min=0.0,
        max=5.0,
        step=0.2,
        description='Scale y with:',
        disabled=False,
        continuous_update=False,
        orientation='vertical',
        readout=True,
        readout_format='.1f',
        layout = widgets.Layout(height='280px', width = '100px')
    )

    out = widgets.interactive_output(draw_scale, {'scalex' : scale_x_slider, 'scaley' : scale_y_slider})
    static = draw_scale(1.0,1.0)
    box = widgets.HBox([scale_y_slider, out])
    vis = widgets.VBox([box, scale_x_slider])
    return display(vis, static)



def draw_cov(iteration):
    data = read_data("gaussian.txt")
    x = [el[0] for el in data]
    y = [el[1] for el in data]
    x = np.array(x)
    y = np.array(y)
    x = x - np.mean(x)
    y = y - np.mean(y)
    
    matrix = np.vstack((x, y)) 
    cov = np.cov(matrix)
    
    
    for i in range(iteration):
        matrix = cov@matrix
    
    plt.scatter(matrix[0], matrix[1], c = 'dodgerblue', marker = 'o', alpha = 0.4)
    plt.title("Dataset multiplied by covariance " + str(iteration) + " times")

def pca_cov():
    iteration_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=5,
        step=1,
        description='Displayed transformations:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        layout = widgets.Layout(width='600px')
    )

    out = widgets.interactive_output(draw_cov, {'iteration' : iteration_slider})
    return display(iteration_slider, out)
