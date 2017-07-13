import notebook
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import *
from IPython.display import display
from sklearn.svm import SVC


def start():
    fit_plot = plt.style.use('ggplot')

    def plot_data(data, labels, sep):
        data_x = data[:, 0]
        data_y = data[:, 1]
        sep_x = sep[:, 0]
        sep_y = sep[:, 1]

        # plot data
        plt.figure(figsize=(4, 4))
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
        # plt.draw(figsize=(4, 4))
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


    # plot approximation
    def plot_approx(plt, clf):
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


    def on_train_info_change(change):
        clf = SVC(C=c.value, kernel='rbf', gamma=gamma.value)

        # fit classifier
        clf.fit(data, labels)

        # plot results
        update_plot_data(fit_plot, data, labels, true_sep)
        plot_approx(clf)


    def on_value_change_sample(change):
        data, labels, true_sep = make_circle_classification_dataset(num_pts=sample_size.value)
        update_plot_data(fit_plot, data, labels, true_sep)

        clf = SVC(C=c.value, kernel='rbf', gamma=gamma.value)

        # fit classifier
        clf.fit(data, labels)

        # plot results
        update_plot_data(fit_plot, data, labels, true_sep)
        plot_approx(clf)

    sample_size = widgets.IntSlider(
        value=50,
        min=50,
        max=1000,
        step=1,
        description='Sample size: ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        slider_color='white'
    )
    split_ratio = widgets.FloatSlider(
        value=0.2,
        min=0,
        max=1.0,
        step=0.1,
        description='Train/Test Split Ratio (0-1): ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        slider_color='white'
    )
    c = widgets.FloatSlider(
        value=0.1,
        min=0.1,
        max=10.0,
        step=0.1,
        description='C: ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        slider_color='white'
    )
    gamma = widgets.FloatSlider(
        value=0.1,
        min=0.1,
        max=1,
        step=0.1,
        description='Gamma: ',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        slider_color='white'
    )

    display(sample_size)

    # init plot
    data, labels, true_sep = make_circle_classification_dataset(num_pts=sample_size.value)

    # preparing the plot

    clf = SVC(C=c.value, kernel='rbf', gamma=gamma.value)

    # fit classifier
    clf.fit(data, labels)

    # plot results
    fit_plot = plot_data(data, labels, true_sep)
    plot_approx(fit_plot, clf)


    sample_size.observe(on_value_change_sample, names='value')

    display(c)
    display(gamma)

    c.observe(on_train_info_change, names='value')
    gamma.observe(on_train_info_change, names='value')