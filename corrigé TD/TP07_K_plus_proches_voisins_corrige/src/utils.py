import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from sklearn.base import BaseEstimator


def _get_regions(predict_fun, xy, shape, model_classes):
    """Return 2D array of classes as integers"""

    # Predict and reshape
    Z_pred = predict_fun(xy).reshape(shape)

    # Turn array of class labels into integers
    cat2num = {cat: num for num, cat in enumerate(model_classes)}
    num2cat = {num: cat for num, cat in enumerate(model_classes)}
    vcat2num = np.vectorize(lambda x: cat2num[x])
    Z_num = vcat2num(Z_pred)

    # Return array of integers in mapping of integers to labels
    return Z_num, num2cat


def _draw_regions(ax, model_classes, num2cat, Z_num):
    """Draw decision regions according to array of integers"""

    # Hack to get colors
    # TODO use legend_out = True
    hdls, hlabels = ax.get_legend_handles_labels()
    hlabels_hdls = {l: h for l, h in zip(hlabels, hdls)}

    # Get mapping of class label to color
    color_dict = {}
    for label in model_classes:
        if str(label) in hlabels_hdls:
            hdl = hlabels_hdls[str(label)]
            color = hdl.get_markerfacecolor()
            color_dict[label] = color
        else:
            raise Exception("No corresponding label found for ", label)

    colors = [color_dict[num2cat[i]] for i in range(len(model_classes))]
    cmap = mpl.colors.ListedColormap(colors)

    ax.imshow(
        Z_num,
        interpolation="nearest",
        extent=ax.get_xlim() + ax.get_ylim(),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        alpha=0.2,
    )


def _draw_boundaries(ax, XX, YY, Z_num, color, model_classes):
    # Boundaries
    mask = np.zeros_like(Z_num, dtype=bool)
    for k in range(len(model_classes) - 1):
        mask |= Z_num == k - 1
        Z_num_mask = np.ma.array(Z_num, mask=mask)
        ax.contour(
            XX,
            YY,
            Z_num_mask,
            levels=[k + 0.5],
            linestyles="dashed",
            corner_mask=True,
            colors=[color],
            antialiased=True,
        )


def _create_grid(ax, resolution):
    # Create grid to evaluate model
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], resolution)
    yy = np.linspace(ylim[0], ylim[1], resolution)
    XX, YY = np.meshgrid(xx, yy)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    return xy, XX, YY


def add_decision_boundary(
    model,
    resolution=100,
    ax=None,
    levels=None,
    label=None,
    color=None,
    region=True,
    model_classes=None,
):
    """Trace une frontière et des régions de décision sur une figure existante.

    :param model: Un modèle scikit-learn ou une fonction `predict`
    :param resolution: La discrétisation en nombre de points par abscisses/ordonnées à utiliser
    :param ax: Les axes sur lesquels dessiner
    :param label: Le nom de la frontière dans la légende
    :param color: La couleur de la frontière
    :param region: Colorer les régions ou pas
    :param model_classes: Les étiquettes des classes dans le cas où `model` est une fonction

    """

    # Set axes
    if ax is None:
        ax = plt.gca()

    if isinstance(model, BaseEstimator):
        model_classes = model.classes_

    if callable(model) and model_classes is None:
        raise Exception("Il faut spécifier le nom des classes dans `model_classes`")

    # Add decision boundary to legend
    color = "red" if color is None else color
    sns.lineplot(x=[0], y=[0], label=label, ax=ax, color=color, linestyle="dashed")

    # Create grid according to current axis and resolution
    xy, XX, YY = _create_grid(ax, resolution)

    if isinstance(model, BaseEstimator):
        fitted_feature_names = getattr(model, "feature_names_in_", None)
        if fitted_feature_names is not None:
            xy_names = pd.DataFrame(xy, columns=fitted_feature_names)
        else:
            xy_names = xy

        if levels is not None:
            if len(model.classes_) != 2:
                raise Exception("Lignes de niveaux supportées avec seulement deux classes")

            # Scikit-learn model, 2 classes + levels
            Z = model.predict_proba(xy_names)[:, 0].reshape(XX.shape)
            Z_num, num2cat = _get_regions(model.predict, xy_names, XX.shape, model_classes)

            # Only 2 classes, simple contour
            ax.contour(
                XX,
                YY,
                Z,
                levels=levels,
                colors=[color]
            )

            _draw_regions(ax, model_classes, num2cat, Z_num)
        else:
            # Scikit-learn model + no levels
            Z_num, num2cat = _get_regions(model.predict, xy_names, XX.shape, model_classes)

            _draw_boundaries(ax, XX, YY, Z_num, color, model_classes)
            if region:
                _draw_regions(ax, model_classes, num2cat, Z_num)
    else:
        if levels is not None:
            raise Exception("Lignes de niveaux avec fonction non supporté")

        # Model is a predict function, no levels
        Z_num, num2cat = _get_regions(model, xy, XX.shape, model_classes)
        _draw_boundaries(ax, XX, YY, Z_num, color, model_classes)
        if region:
            _draw_regions(ax, model_classes, num2cat, Z_num)
