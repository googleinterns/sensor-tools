import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA


def scatter_PCA(X, Y, components, alpha):
    """
       Description: Creates PCA scatter plot where X is a numpy array of samples and Y contains the corresponding labels. 

       Args:
           X -- numpy array (Numpy array of data to be plotted)
           Y -- numpy array (Numpy array with labels for data in X)
           components -- int (Number of features of data in X) 
           alpha -- double (From [0.0 - 1.0], level of opacity for the dots on the plot)

     """
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(X)
    scatter_plot(pca_result, Y, alpha)


def scatter_ICA(X, Y, components, alpha):
    """
       Description: Creates ICA scatter plot where X is a numpy array of samples and Y contains the corresponding labels. 

       Args:
           X -- numpy array (Numpy array of data to be plotted)
           Y -- numpy array (Numpy array with labels for data in X)
           components -- int (Number of features of data in X) 
           alpha -- double (From [0.0 - 1.0], level of opacity for the dots on the plot)

    """
    ica = FastICA(n_components=components)
    ica_result = ica.fit_transform(X)
    scatter_plot(ica_result, Y, alpha)


def scatter_TSNE(X, Y, components, alpha):
    """
       Description: Creates t-SNE scatter plot where X is a numpy array of samples and Y contains the corresponding labels. 

       Args:
           X -- numpy array (Numpy array of data to be plotted)
           Y -- numpy array (Numpy array with labels for data in X)
           components -- int (Number of features of data in X) 
           alpha -- double (From [0.0 - 1.0], level of opacity for the dots on the plot)

    """
    RS = 20150101
    TSNE_proj = TSNE(random_state=RS, n_components=components).fit_transform(X)
    scatter_plot(TSNE_proj, Y, alpha)


def scatter_plot(result, Y, alpha):
    """
       Description: Creates scatter plot from output of PCA, ICA, t-SNE functions 

       Args:
           result -- numpy array (nshape = (n_samples, n_components) Embedding of the training data in low-dimensional space)
           Y -- numpy array (Numpy array with labels for data in X)
           alpha -- double (From [0.0 - 1.0], level of opacity for the dots on the plot)

    """
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 1.25})
    df_subset = {"1": [], "2": []}
    df_subset['1'] = result[:, 0]
    df_subset['2'] = result[:, 1]
    df_subset['y'] = Y
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x="2", y="1",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_subset,
        legend="full",
        alpha=alpha
    )
