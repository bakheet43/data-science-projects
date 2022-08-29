import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import tensorflow as tf
from keras import Sequential, Input
from keras.layers import Dense
from matplotlib import gridspec
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from statsmodels import robust
from statsmodels.sandbox.distributions.examples.ex_mvelliptical import fig

sns.set_style("darkgrid")
data = load_iris()
cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
feature_set = pd.DataFrame(data.data, columns=cols)
species = pd.Series(data.target, name="species").map({
    0: 'Iris-setosa',
    1: 'Iris-versicolor',
    2: 'Iris-virginica'
})
iris_data = pd.merge(feature_set, species, left_index=True, right_index=True)


def main_data():
    # print a concise summary of a DataFrame.
    print(iris_data.info())
    # Generate descriptive statistics.
    # Descriptive statistics include those that summarize the central tendency,
    # dispersion and shape of a dataset’s distribution, excluding NaN values
    print(iris_data.describe())
    # check class distribution
    _ = sns.countplot(iris_data.species)
    _ = plt.title("Class distribution => Balanced Dataset", fontsize=14)


def univariate_analysis():
    iris_setosa = iris_data.loc[iris_data.species == "Iris-setosa"]
    iris_versicolor = iris_data.loc[iris_data.species == "Iris-versicolor"]
    iris_virginica = iris_data.loc[iris_data.species == "Iris-virginica"]
    print("Setosa mean: ", np.mean(iris_setosa['petal_length']))
    print("Setosa corrupted mean: ", np.mean(np.append(iris_setosa['petal_length'], 50)))
    print("versicolor mean: ", np.mean(iris_versicolor['petal_length']))
    print("virginica mean: ", np.mean(iris_virginica['petal_length']))
    print()
    print("Setosa variance: ", np.var(iris_setosa['petal_length']))
    print("Setosa corrupted variance: ", np.var(np.append(iris_setosa['petal_length'], 50)))
    print("versicolor variance: ", np.var(iris_versicolor['petal_length']))
    print("virginica variance: ", np.var(iris_virginica['petal_length']))
    print()
    print("Setosa std: ", np.std(iris_setosa['petal_length']))
    print("Setosa corrupted std: ", np.std(np.append(iris_setosa['petal_length'], 50)))
    print("versicolor std: ", np.std(iris_versicolor['petal_length']))
    print("virginica std: ", np.std(iris_virginica['petal_length']))
    print("Median: ")
    print("Setosa Median: ", np.median(iris_setosa['petal_length']))
    print("Setosa corrupted Median: ", np.median(np.append(iris_setosa['petal_length'], 50)))
    print("versicolor Median: ", np.median(iris_versicolor['petal_length']))
    print("virginica Median: ", np.median(iris_virginica['petal_length']))

    print("\nQuantiles: [0, 25, 50, 75]")
    print("Setosa Quantile: ", np.percentile(iris_setosa['petal_length'], np.arange(0, 100, 25)))
    print("Setosa corrupted Quantile: ",
          np.percentile(np.append(iris_setosa['petal_length'], 50), np.arange(0, 100, 25)))
    print("versicolor Quantile: ", np.percentile(iris_versicolor['petal_length'], np.arange(0, 100, 25)))
    print("virginica Quantile: ", np.percentile(iris_virginica['petal_length'], np.arange(0, 100, 25)))

    print("\n90th Percentiles")
    print("Setosa Percentile: ", np.percentile(iris_setosa['petal_length'], 90))
    print("Setosa corrupted Percentile: ", np.percentile(np.append(iris_setosa['petal_length'], 50), 90))
    print("versicolor Percentile: ", np.percentile(iris_versicolor['petal_length'], 90))
    print("virginica Percentile: ", np.percentile(iris_virginica['petal_length'], 90))
    print("\nMedian Absolute Deviation")
    print("Setosa MAD: ", robust.mad(iris_setosa['petal_length']))
    print("Setosa corrupted MAD: ", robust.mad(np.append(iris_setosa['petal_length'], 50)))
    print("versicolor MAD: ", robust.mad(iris_versicolor['petal_length']))
    print("virginica MAD: ", robust.mad(iris_virginica['petal_length']))


def univariate_analysis_shape():
    vfig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for no, column in enumerate(iris_data.columns[:-1], 1):
        ax = fig.add_subplot(2, 2, no)
        sns.distplot(iris_data.loc[iris_data.species == 'Iris-setosa', f"{column}"], label="Setosa")
        sns.distplot(iris_data.loc[iris_data.species == 'Iris-versicolor', f"{column}"], label="Versicolor")
        sns.distplot(iris_data.loc[iris_data.species == 'Iris-virginica', f"{column}"], label="Virginica")
        ax.legend()
        plt.tight_layout(pad=2.0)
        plt.show()


def pdf_cdf():
    counts, bin_edges = np.histogram(iris_data.loc[iris_data.species == 'Iris-setosa', 'petal_length'],
                                     bins=10, density=True)

    pdf = counts / sum(counts)
    print("PDF: ", pdf)
    print("CDF: ", bin_edges)

    # cdf
    cdf = np.cumsum(pdf)
    _ = plt.plot(bin_edges[1:], pdf)
    _ = plt.plot(bin_edges[1:], cdf)
    fig = plt.figure(figsize=(10, 6))

    for i, cls in enumerate(iris_data.species.unique(), 1):
        counts, bin_edges = np.histogram(iris_data.loc[iris_data.species == f'{cls}', 'petal_length'],
                                         bins=10, density=True)

        pdf = counts / sum(counts)
        # print("PDF: ", pdf)
        # print("CDF: ", bin_edges)

        # cdf
        cdf = np.cumsum(pdf)
        _ = plt.plot(bin_edges[1:], pdf)
        _ = plt.plot(bin_edges[1:], cdf, label=f'{cls}')

    plt.title(f"{cls}: PDF & CDF plot")
    plt.xlabel("petal_length")
    plt.legend()
    plt.show()


def cumulative_density():
    fig = plt.figure(figsize=(9, 40))
    outer = gridspec.GridSpec(4, 1, wspace=0.2, hspace=0.2)

    for i, col in enumerate(iris_data.columns[:-1]):
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                 subplot_spec=outer[i], wspace=0.2, hspace=0.4)

        ax = plt.Subplot(fig, inner[0])
        _ = sns.boxplot(y="species", x=f"{col}", data=iris_data, ax=ax)
        _ = sns.stripplot(y="species", x=f"{col}", data=iris_data, jitter=True, dodge=True, linewidth=1, ax=ax)
        _ = ax.set_title("Box Plot")
        fig.add_subplot(ax)

        ax = plt.Subplot(fig, inner[1])
        _ = sns.violinplot(y="species", x=f"{col}", data=iris_data, inner='quartile', ax=ax)
        # _ = sns.stripplot(x="species", y="petal_length", data=iris_data, jitter=True, dodge=True, linewidth=1, ax=ax)
        _ = ax.set_title("Violin Plot")
        fig.add_subplot(ax)
    fig.show()
    plt.figure(figsize=(15, 10))
    for i, j in enumerate(iris_data.columns[:-1], 1):
        plt.subplot(2, 2, i)
        _ = sns.boxplot(x="species", y=f"{j}", data=iris_data)
        _ = sns.stripplot(x="species", y=f"{j}", data=iris_data, jitter=True, dodge=True, linewidth=1)
        _ = plt.title("Box Plot")

    plt.figure(figsize=(15, 10))
    for i, j in enumerate(iris_data.columns[:-1], 1):
        plt.subplot(2, 2, i)
        _ = sns.violinplot(y="species", x=f"{col}", data=iris_data, inner='quartile')
        # _ = sns.stripplot(x="species", y="petal_length", data=iris_data, jitter=True, dodge=True, linewidth=1, ax=ax)
        _ = plt.title("Violin Plot")
    plt.tight_layout(pad=2)
    iris_data.sample(frac=1)

    random_idx = np.random.choice(range(0, 150), 20)
    print(simple_rule(iris_data.sample(frac=1).iloc[random_idx]))


def simple_rule(subset):
    cls = []
    for idx, row in subset.iterrows():
        if row['petal_length'] <= 2:
            cls.append("Iris-setosa")
        elif 2 < row['petal_length'] <= 4.6:
            cls.append("Iris-versicolor")
        else:
            cls.append("Iris-virginica")
    # accuracy
    cls = np.array(cls)

    return accuracy_score(cls, subset.species.values)


def correlation_plot():
    plt.figure(figsize=(8, 6))
    _ = sns.heatmap(iris_data.corr(), vmin=-1, vmax=1, annot=True, cmap='afmhot')
    _ = sns.relplot(x='petal_length', y='petal_width', hue='species', data=iris_data, height=7)
    _ = plt.title("Scatter plot", fontsize=14)
    g = sns.jointplot(x="sepal_length", y="petal_width", data=iris_data, kind="kde")
    g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
    g.ax_joint.collections[0].set_alpha(0)
    g.set_axis_labels("$Sepal$ $length$", "$Petal$ $width$");

    p = sns.jointplot(x="sepal_length", y="petal_length", data=iris_data, kind="kde")
    p.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
    p.ax_joint.collections[0].set_alpha(0)
    p.set_axis_labels("$Sepal$ $length$", "$Petal$ $length$")
    _ = plt.figure(figsize=(15, 10))
    _ = sns.pairplot(iris_data, hue="species", height=3, diag_kind="kde")


def countour_probablity():
    iris_setosa = iris_data.loc[iris_data.species == "Iris-setosa"]
    g = sns.jointplot(x="petal_length", y="petal_width", data=iris_setosa, kind="kde")
    g.plot_joint(plt.scatter, c="k", s=30, linewidth=1, marker="+")
    g.ax_joint.collections[0].set_alpha(0)
    g.set_axis_labels("$Petal$ $length$", "$Petal$ width$")


def multi_variate():
    df = px.data.iris()
    fig = px.scatter_3d(iris_data, x='petal_length', y='petal_width', z='sepal_length',
                        color='species')
    fig.show()


def iris_versicolor_virginica():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, j in iris_data.groupby("species"):
        temp = j[['sepal_length', 'sepal_width', 'petal_length']]
        ax.scatter(temp['sepal_length'], temp['sepal_width'], \
                   temp['petal_length'], s=40, edgecolor='k')

    ax.set_xlabel("sepal_length")
    ax.set_ylabel("sepal_width")
    ax.set_zlabel("petal_length")
    plt.title("3D plot to check for seperation")
    plt.show()


def model_building():
    X = iris_data.drop(['species'], axis=1)
    y = iris_data['species'].map({
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    })
    print(X.shape)
    print(y.shape)
    models = []

    models.append(("LogisticRegression", LogisticRegression(max_iter=1000)))
    models.append(("SVC", SVC(kernel="rbf", gamma=5, C=0.001, max_iter=1000)))

    models.append(("KNeighbors", KNeighborsClassifier(n_neighbors=12)))
    models.append(("DecisionTree", DecisionTreeClassifier()))
    models.append(("RandomForest", RandomForestClassifier()))
    rf2 = RandomForestClassifier(n_estimators=100, criterion='gini',
                                 max_depth=10, random_state=42, max_features=None)
    models.append(("RandomForest2", rf2))
    models.append(('NB', GaussianNB()))
    models.append(("MLPClassifier",
                   MLPClassifier(hidden_layer_sizes=(10, 10), solver='adam', max_iter=2000, learning_rate='adaptive',
                                 random_state=42)))
    # naive feature selection
    for i in range(1, 5):
        cols = X.columns[:i]
        X_temp = X[cols].values
        results = []
        names = []
        for name, model in models:
            try:
                result = cross_val_score(model, X[cols], y, cv=5, scoring='accuracy')
            except:
                result = cross_val_score(model, X[cols].reshape(-1, 1), y, cv=5, scoring='accuracy')
            names.append(name)
            results.append(result)
        print(f"Using features: {cols}")
        for i in range(len(names)):
            # f"{'1':0>8}
            print(f"Algo: {names[i]}, Result: {round(results[i].mean(), 2)}")
        print()


def petal_seperation():
    X = iris_data.drop(['species'], axis=1)
    y = iris_data['species'].map({
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    })
    models = []
    single_feature_models = models[:]
    single_feature_models.pop(2)
    single_feature_models.insert(2, ("KNeighbors", KNeighborsClassifier(n_neighbors=3)))

    two_feature_models = models[:]
    two_feature_models.pop(2)
    two_feature_models.insert(2, ("KNeighbors", KNeighborsClassifier(n_neighbors=5)))

    X_selected_1 = X[['petal_length']].values
    X_selected_2 = X[['petal_length', 'petal_width']].values

    X_ = [X_selected_1, X_selected_2]
    y = y.ravel()

    mods = [single_feature_models, two_feature_models]
    for i in range(2):
        curr_models = mods[i]
        names = []
        results = []
        for name, mod in curr_models:
            if i == 0:
                result = cross_val_score(mod, X_selected_1.reshape(-1, 1), y, cv=5, scoring='accuracy')
            else:
                result = cross_val_score(mod, X_selected_2, y, cv=5, scoring='accuracy')

            names.append(name)
            results.append(result)

        print(f"Features: {X_[i].shape[1]}")
        for j in range(len(names)):
            print(f"Algo: {names[j]}, Result: {round(results[j].mean(), 2)}")
        print()


def feature_selection_tensorflow():
    X = iris_data.drop(['species'], axis=1)
    y = iris_data['species'].map({
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    })
    x_train, x_val, y_train, y_val = train_test_split(X, y, shuffle=True, stratify=y,
                                                      random_state=42, test_size=0.1)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(32)
    train_dataset = train_dataset.shuffle(150, reshuffle_each_iteration=True)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(32)
    val_dataset = val_dataset.shuffle(15, reshuffle_each_iteration=True)
    print(val_dataset)
    model = Sequential([
        Input(shape=(4,)),
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dense(3, activation='softmax'),
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])
    history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, verbose=0)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 101), history.history['loss'], label="Loss")
    plt.plot(range(1, 101), history.history['val_loss'], label="validation_loss")
    plt.legend()
    plt.title("Epoch Vs. loss")
    plt.xlabel("Epoch")
    plt.ylabel("loss")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, 101), history.history['accuracy'], label="accuracy")
    plt.plot(range(1, 101), history.history['val_accuracy'], label="validation_accuracy")
    plt.legend()
    plt.title("Epoch Vs. accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")
    plt.show()


def feature_extraction():
    # Seperating X and y
    X = iris_data.drop(['species'], axis=1)
    y = iris_data['species'].map({
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    })
    print(X.shape)
    print(y.shape)
    pca = PCA(n_components=0.95)
    X_transformed = pca.fit_transform(X)
    print(X_transformed.shape)
    print(pca.n_components_)
    print(pca.explained_variance_ratio_)
    x_train, x_val, y_train, y_val = train_test_split(X_transformed, y, shuffle=True, stratify=y, test_size=0.1)

    log = LogisticRegression(max_iter=500)
    log.fit(x_train, y_train)
    pred = log.predict(x_val)
    accuracy_score(y_val, pred)


if __name__ == '__main__':
    feature_extraction()
