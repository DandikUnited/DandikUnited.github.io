### in this cell we import necessary libraries for the demo
import numpy as np  # a fundamental numerical linear algebra library
import matplotlib.pyplot as plt  # a basic plotting library
import pandas as pd
from sklearn.metrics import accuracy_score


# make a toy circle dataset
def make_line_classification_dataset(num_pts):
    '''
    This function generates a random line dataset with two classes. -
    You can run this a couple times to get a distribution you like visually.  
    You can also adjust the num_pts parameter to change the total number of points in the dataset.
    '''

    # generate points
    num_misclass = 5  # total number of misclassified points
    data_x = 4 * np.random.rand(num_pts) - 2
    data_y = 4 * np.random.rand(num_pts) - 2
    data_x.shape = (len(data_x), 1)
    data_y.shape = (len(data_y), 1)
    data = np.concatenate((data_x, data_y), axis=1)

    # make separator
    x_f = 4 * np.linspace(0, 1, 100) - 2
    m, b = np.random.randn(2, 1)
    y_f = m * x_f + b
    x_f.shape = (len(x_f), 1)
    y_f.shape = (len(y_f), 1)
    sep = np.concatenate((x_f, y_f), axis=1)

    # make labels and flip a few to show some misclassifications
    labels = m * data_x + b
    ind1 = np.argwhere(data_y > labels)
    ind1 = [v[0] for v in ind1]
    ind2 = np.argwhere(data_y <= labels)
    ind2 = [v[0] for v in ind2]
    labels[ind1] = -1
    labels[ind2] = +1

    flip = np.random.permutation(num_pts)
    flip = flip[:num_misclass]
    for i in flip:
        labels[i] = (-1) * labels[i]

    # plot everything
    plot_data(data, labels, sep)

    # return datapoints and labels for further
    return data, labels, sep


# make a toy circle dataset
def make_circle_classification_dataset(num_pts):
    '''
    This function generates a random circle dataset with two classes. 
    You can run this a couple times to get a distribution you like visually.  
    You can also adjust the num_pts parameter to change the total number of points in the dataset.
    '''

    # generate points
    num_misclass = 5  # total number of misclassified points
    s = np.random.rand(num_pts)
    data_x = np.cos(2 * np.pi * s)
    data_y = np.sin(2 * np.pi * s)
    radi = 2 * np.random.rand(num_pts)
    data_x = data_x * radi
    data_y = data_y * radi
    data_x.shape = (len(data_x), 1)
    data_y.shape = (len(data_y), 1)
    data = np.concatenate((data_x, data_y), axis=1)

    # make separator
    s = np.linspace(0, 1, 100)
    x_f = np.cos(2 * np.pi * s)
    y_f = np.sin(2 * np.pi * s)
    x_f.shape = (len(x_f), 1)
    y_f.shape = (len(y_f), 1)
    sep = np.concatenate((x_f, y_f), axis=1)

    # make labels and flip a few to show some misclassifications
    labels = radi.copy()
    ind1 = np.argwhere(labels > 1)
    ind1 = [v[0] for v in ind1]
    ind2 = np.argwhere(labels <= 1)
    ind2 = [v[0] for v in ind2]
    labels[ind1] = -1
    labels[ind2] = +1

    flip = np.random.permutation(num_pts)
    flip = flip[:num_misclass]
    for i in flip:
        labels[i] = (-1) * labels[i]

    # return datapoints and labels for study
    return data, labels, sep


# function - plot data with underlying target function generated in the previous Python cell
def plot_data(data, labels, sep):
    data_x = data[:, 0]
    data_y = data[:, 1]
    sep_x = sep[:, 0]
    sep_y = sep[:, 1]

    # plot data
    fig = plt.figure(figsize=(4, 4))
    pos_inds = np.argwhere(labels == 1)
    pos_inds = [s[0] for s in pos_inds]

    neg_inds = np.argwhere(labels == -1)
    neg_inds = [s[0] for s in neg_inds]
    plt.scatter(data_x[pos_inds], data_y[pos_inds], color='b', linewidth=1, marker='o', edgecolor='k', s=50)
    plt.scatter(data_x[neg_inds], data_y[neg_inds], color='r', linewidth=1, marker='o', edgecolor='k', s=50)

    # plot target
    plt.plot(sep_x, sep_y, '--k', linewidth=3)

    # clean up plot
    plt.yticks([], [])
    plt.xlim([-2.1, 2.1])
    plt.ylim([-2.1, 2.1])
    plt.axis('off')
    return plt


def update_plot_data(plt, data, labels, sep):
    plt.cla()
    plt.clf()
    data_x = data[:, 0]
    data_y = data[:, 1]
    sep_x = sep[:, 0]
    sep_y = sep[:, 1]

    # plot data
    #plt.draw(figsize=(4, 4))
    pos_inds = np.argwhere(labels == 1)
    pos_inds = [s[0] for s in pos_inds]

    neg_inds = np.argwhere(labels == -1)
    neg_inds = [s[0] for s in neg_inds]
    plt.scatter(data_x[pos_inds], data_y[pos_inds], color='b', linewidth=1, marker='o', edgecolor='k', s=50)
    plt.scatter(data_x[neg_inds], data_y[neg_inds], color='r', linewidth=1, marker='o', edgecolor='k', s=50)

    # plot target
    plt.plot(sep_x, sep_y, '--k', linewidth=3)

    # clean up plot
    plt.yticks([], [])
    plt.xlim([-2.1, 2.1])
    plt.ylim([-2.1, 2.1])
    plt.axis('off')



# function - plot training and testing sets
def plot_train_test(data_train, data_test, labels_train, labels_test):
    # plot data
    fig = plt.figure(figsize=(4, 4))
    ind0 = np.argwhere(labels_train == -1)
    ind0 = [v[0] for v in ind0]
    ind1 = np.argwhere(labels_train == 1)
    ind1 = [v[0] for v in ind1]

    plt.scatter(data_train[ind0, 0], data_train[ind0, 1], color='b', linewidth=2, marker='o')
    plt.scatter(data_train[ind1, 0], data_train[ind1, 1], color='r', linewidth=2, marker='o')

    ind0 = np.argwhere(labels_test == -1)
    ind0 = [v[0] for v in ind0]
    ind1 = np.argwhere(labels_test == 1)
    ind1 = [v[0] for v in ind1]

    plt.scatter(data_test[ind0, 0], data_test[ind0, 1], color='b', linewidth=1, marker='s', edgecolor='k', s=50)
    plt.scatter(data_test[ind1, 0], data_test[ind1, 1], color='r', linewidth=1, marker='s', edgecolor='k', s=50)

    # clean up plot
    plt.yticks([], [])
    plt.xlim([-2.1, 2.1])
    plt.ylim([-2.1, 2.1])
    plt.axis('off')
    plt.title('training (circles) and testing (squares) sets')


# plot approximation
def plot_approx(clf):
    # plot classification boundary and color regions appropriately
    r = np.linspace(-2.1, 2.1, 500)
    s, t = np.meshgrid(r, r)
    s = np.reshape(s, (np.size(s), 1))
    t = np.reshape(t, (np.size(t), 1))
    h = np.concatenate((s, t), 1)

    # use classifier to make predictions
    z = clf.predict(h)

    # reshape predictions for plotting
    s.shape = (np.size(r), np.size(r))
    t.shape = (np.size(r), np.size(r))
    z.shape = (np.size(r), np.size(r))

    # show the filled in predicted-regions of the plane
    plt.contourf(s, t, z, colors=['r', 'b'], alpha=0.2, levels=range(-1, 2))

    # show the classification boundary if it exists
    if len(np.unique(z)) > 1:
        plt.contour(s, t, z, colors='k', linewidths=3)


def update_plot_approx(plt, clf):
    # plot classification boundary and color regions appropriately
    r = np.linspace(-2.1, 2.1, 500)
    s, t = np.meshgrid(r, r)
    s = np.reshape(s, (np.size(s), 1))
    t = np.reshape(t, (np.size(t), 1))
    h = np.concatenate((s, t), 1)

    # use classifier to make predictions
    z = clf.predict(h)

    # reshape predictions for plotting
    s.shape = (np.size(r), np.size(r))
    t.shape = (np.size(r), np.size(r))
    z.shape = (np.size(r), np.size(r))
    plt.cla()
    plt.clf()

    # show the filled in predicted-regions of the plane
    plt.contourf(s, t, z, colors=['r', 'b'], alpha=0.2, levels=range(-1, 2))

    # show the classification boundary if it exists
    if len(np.unique(z)) > 1:
        plt.contour(s, t, z, colors='k', linewidths=3)


# plot training and testing errors
def plot_cv_scores(train_errors, test_errors, param_range):
    # plot training and testing errors
    plt.plot(param_range, train_errors, marker='o', color=[1, 0.8, 0.5])
    plt.plot(param_range, test_errors, marker='o', color=[0, 0.7, 1])

    # clean up plot
    plt.xlim([min(param_range) - 0.3, max(param_range) + 0.3])
    plt.ylim([min(min(train_errors), min(test_errors)) - 0.05, max(max(train_errors), max(test_errors)) + 0.05]);
    plt.xlabel('parameter values')
    plt.ylabel('error')
    plt.xticks(param_range);
    plt.title('cross validation errors', fontsize=14)
    plt.legend(['training error', 'testing error'], loc='center left', bbox_to_anchor=(1, 0.5))