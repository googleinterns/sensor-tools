import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA

def scatter_PCA(X, Y, components, alpha):
    pca = PCA(n_components=components)
    pca_result = pca.fit_transform(X)
    scatter_plot(pca_result, Y, alpha)


def scatter_ICA(X, Y, components, alpha):
    ica = FastICA(n_components=components)
    ica_result = ica.fit_transform(X)
    scatter_plot(ica_result, Y, alpha)


def scatter_TSNE(X, Y, components, alpha):
    RS = 20150101
    TSNE_proj = TSNE(random_state=RS, n_components=components).fit_transform(X)
    scatter_plot(TSNE_proj, Y, alpha)


def scatter_plot(result, Y, alpha):
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 1.25})
    df_subset = {"negative": [], "positive": []}
    df_subset['positive'] = result[:, 0]
    df_subset['negative'] = result[:, 1]
    df_subset['y'] = Y
    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x="negative", y="positive",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df_subset,
        legend="full",
        alpha=alpha
    )