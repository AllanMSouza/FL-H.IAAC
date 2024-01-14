import sys
import numpy as np

from quanttree.cdm_src import CDM_QT_EWMA
from quanttree.qtewma_src import QT_EWMA

VISUALIZE_RESULTS = False  # requires matplotlib

"""
In this demo, we demonstrate the usage of CDM [1] individually monitoring two classes in a stream of data, and we
compare it against QT-EWMA [2], which monitors the distribution of the whole datastream.

The stationary distribution consists of two 1-dimensional uniform distributions, namely, U(0,1) and U(-0.5, 0.5).
After the change point `tau`, only the first class distribution drifts and becomes U(-0.5, 0.5). 

We plot the test statistic and the detection thresholds of the considered methods.

Parameters
----------
seed : int
    A seed to be fed to numpy for experimental reproducibility
training_points_per_class : int
    The number of training samples drawn from each stationary distribution
tau : int
    The index of the change point
points_after_tau : int
    The number of post-change samples generated after tau
ARL_0 : int
    The target Average Run Length (ARL_0), namely, the number of stationary samples monitored before a false alarm
lam : float
    The weight assigned by QT-EWMA to the incoming samples.
K : int
    The number of bins constructed by the QuantTree algorithm

References
----------
[1] "Class Distribution Monitoring for Concept Drift Detection"
D. Stucchi, L. Frittoli, G. Boracchi, International Joint Conference on Neural Networks (IJCNN), 2022.

[2] "Change Detection in Multivariate Datastreams Controlling False Alarms"
L. Frittoli, D. Carrera, G. Boracchi, Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD), 2021.
"""


def get_stationary(npointsperclass: int, do_shuffle: bool = True):
    _points = np.concatenate([
        # np.random.uniform(low=0.25, high=1, size=npointsperclass),
        np.random.uniform(low=0, high=1, size=npointsperclass),
        np.random.uniform(low=-.5, high=.5, size=npointsperclass),
        np.random.uniform(low=-.5, high=.5, size=npointsperclass),
        np.random.uniform(low=-.5, high=.5, size=npointsperclass),
    ])
    _labels = np.concatenate([np.random.randint(2, 3, npointsperclass), np.random.randint(1, 2, npointsperclass), np.random.randint(0, 1, npointsperclass), np.random.randint(3, 4, npointsperclass)])
    unique = np.unique(_labels)
    print("unicos: ", unique)
    train_stream = {i: _points[_labels == i] for i in np.unique(_labels)}
    for i in np.unique(_labels):
        print("tamanho classe: ", i, " : ", train_stream[i].shape[0])
    # assert all([train_stream[i].shape[0] >= 512 for i in _labels]), \
    #     "[QT_EWMA] Not enough points to train a QuantTree aaa"

    if do_shuffle:
        idxs = np.arange(4 * npointsperclass)
        np.random.shuffle(idxs)
        _points = _points[idxs]
        _labels = _labels[idxs]

    return _points, _labels

def shuffle(_points, _labels):
    idxs = np.arange(len(_labels))
    np.random.shuffle(idxs)
    _points = _points[idxs]
    _labels = _labels[idxs]

    return _points, _labels


def get_postchange(npointsperclass: int, do_shuffle: bool = True):
    _points = np.concatenate([
        # np.random.uniform(low=-1, high=-.25, size=npointsperclass),
        np.random.uniform(low=-.5, high=.5, size=npointsperclass),
        np.random.uniform(low=-.5, high=.5, size=npointsperclass),
    ])
    n = npointsperclass//2
    _labels = np.concatenate([np.zeros(npointsperclass), np.array([3]*n), np.array([2]*n)])

    if do_shuffle:
        idxs = np.arange(2 * npointsperclass)
        np.random.shuffle(idxs)
        _points = _points[idxs]
        _labels = _labels[idxs]

    return _points, _labels

def upsample(training_data, training_labels, min_samples, n_classes):

    try:

        counts = np.unique(training_labels, return_counts=True)
        remaining_samples_per_class = {i: min_samples for i in range(n_classes)}
        for i, j in zip(counts[0], counts[1]):
            remaining_samples_per_class[i] = max(0, min_samples - j)
        print("falta: ", remaining_samples_per_class)

        # raise

        for class_ in remaining_samples_per_class.keys():

            print("ol", training_data.shape, np.array(training_labels == class_).shape)
            r = np.take(training_data, np.array(training_labels == class_), axis=0)
            remaining = int(remaining_samples_per_class[class_])
            if remaining > 0:

                print("te", r.shape)
                if len(r) == 0:

                    # mean = np.mean(training_data, axis=0)
                    mean = np.zeros(training_data.shape)
                    data = []
                    for i in range(remaining):
                        data.append(mean)
                    print("p1")
                elif len(r) == 1:
                    data = np.array(list(r[0]) * remaining)
                    print("p2", r.shape, training_data.shape)
                else:
                    print("p3", r.shape)
                    idx = np.random.choice(range(len(r)), size=remaining)
                    data = np.take(r, idx, axis=0)
                    print(data.shape)




                print("dw", data.shape)

                training_data = np.concatenate([training_data, data])
                training_labels = np.concatenate([training_labels, np.array([class_]*len(data))])

        print("unn: ", np.unique(training_labels, return_counts=True))

        return training_data, training_labels

    except Exception as e:
        print("quan cdm")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def quan(stream, labels, training_data_, training_labels_, n_classes, r):

    try:
        # --- Demo parameters
        seed = 2020                         # seed for experiment reproducibility
        if seed is not None:
            np.random.seed(seed)

        training_points_per_class = 32   # number of training points
        tau = len(training_data_)                           # change point
        points_after_tau = n_classes * training_points_per_class             # length of the post-change datastream
        ARL_0 = 1000                        # target Average Run Length
        lam = 0.03                          # weight of incoming samples in QT-EWMA
        K = 32                              # number of histogram bins

        sh1 = training_data_.shape[1]
        # training_labels = []
        # n_classes = 12
        # for i in range(n_classes):
        #     training_labels += [i] * training_points_per_class
        #
        # training_labels = np.array(training_labels)
        #
        # training_data = []
        #
        # for i in range(n_classes):
        #     if len(training_data) == 0:
        #         training_data = np.random.uniform(low=0, high=2, size=(training_points_per_class, sh1))
        #     else:
        #         training_data = np.concatenate(
        #             [training_data, np.random.uniform(low=0, high=2, size=(training_points_per_class, sh1))])
        #
        # unique = np.unique(training_labels)
        #
        # training_data = np.concatenate([training_data, training_data_])
        # training_labels = np.concatenate([training_labels, training_labels_])

        training_data, training_labels = upsample(training_data_, training_labels_, training_points_per_class, n_classes)

        training_data, training_labels = shuffle(training_data, training_labels)

        print("unicos 2: ", np.unique(training_labels, return_counts=True))
        # Generate stationary data
        # tem que ter todas as classes

        stationary_labels = []
        n_classes = 12
        for i in range(n_classes):
            stationary_labels += [i] * tau

        stationary_labels = np.array(stationary_labels)

        stationary_data = []

        for i in range(n_classes):
            if len(stationary_data) == 0:
                stationary_data = np.random.uniform(low=0, high=1, size=(tau, sh1))
            else:
                stationary_data = np.concatenate(
                    [stationary_data, np.random.uniform(low=0, high=1, size=(tau, sh1))])

        stationary_data, stationary_labels = shuffle(stationary_data, stationary_labels)

        # Generate post-change data
        # postchange_labels = []
        # n_classes = 12
        # for i in range(n_classes):
        #     postchange_labels += [i] * (points_after_tau // 2)

        postchange_data = np.random.uniform(0, 1, size=(len(labels), sh1))

        postchange_data, labels = shuffle(postchange_data, labels)

        # Concatenating in a datastream
        print("tamanho dados originais: ", stationary_data.shape, stationary_labels.shape)
        stream = np.concatenate([stationary_data, postchange_data])
        labels = np.concatenate([stationary_labels, labels])

        print("stream: ", stream.shape)
        print("labels: ", labels.shape, np.unique(labels, return_counts=True))

        # --- Training and monitoring

        # CDM (w/ QT-EWMA)
        cdm = CDM_QT_EWMA(nqtree=training_points_per_class, ARL_0=ARL_0, lam=lam, K=K)
        # Training (separately trains two QT-EWMA on the individual classes)

        cdm.train(train_points=training_data, train_labels=training_labels)
        # Independently monitors the streams
        cdm_tau_hat = cdm.monitor(stream=stream, labels=labels)
        print("tau hat: ", cdm_tau_hat)
        print("a: ", cdm.labels_list, cdm.nclasses)
        cdm_statistics = cdm.compute_statistics(stream=stream, labels=labels)
        # cdm_thresholds = cdm.get_thresholds(streams_length=[stream[labels == i].shape[0] for i in [0, 1]])
        cdm_changed_class = labels[cdm_tau_hat]

        print("Round: ", r)
        if cdm_tau_hat == -1:
            print("CDM did not detect any change")
            return -1
        else:
            if cdm_changed_class == 1:
                print(f"CDM detected a change at {cdm_tau_hat} over class {int(cdm_changed_class)}")
                print(f"Since {int(cdm_changed_class)} has not changed, it is a false alarm.")
            else:
                print(f"CDM detected a change at {cdm_tau_hat} over class {int(cdm_changed_class)}")
                if cdm_tau_hat < tau:
                    print(f"Since {cdm_tau_hat} < {tau}, it is a false alarm!")

            return cdm_tau_hat

    except Exception as e:
        print("quan cdm")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)