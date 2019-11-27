import itertools
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from typing import List, Tuple, Dict
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster import cluster_visualizer
from utils.annotation_interfaces import DetectionAnnotations


class BBoxDistributionAnalyzer:

    def __init__(self, anno: DetectionAnnotations, distance_measure: str = 'l2'):
        """

        Args:
            anno (DetectionAnnotations): DetectionAnnotations object
            distance_measure (str): similarity measure
        """

        classes_list = self._collect_classes(anno)
        self.anno = anno
        self.classes_list = classes_list
        self.bbox_data = self._collect_bbox(anno, classes_list)
        self.metric = None
        self.distance_measure = distance_measure
        self.similarity_matrix = None
        self.analysis_result = None

        if distance_measure == "l1":
            self.metric = self._manhattan_distance
        elif distance_measure == "l2":
            self.metric = self._euclidean_distance
        elif distance_measure == "inner_product":
            self.metric = self._inner_product
        elif distance_measure == "cosine_similarity":
            self.metric = self._cosine_similarity
        else:
            raise RuntimeError("Not supported {} metric".format(distance_measure))

    def fit(self):
        self.analysis_result = self._distribution_modeling(self.classes_list, self.bbox_data)
        self.similarity_matrix = self._similarity_intra_classes(self.analysis_result)

    def show_similarity_matrix(self, is_save: bool = False):
        """
        Display and save figure about similarity matrix

        Args:
            (bool) is_save : if True, save figure
                                False, not save figure

        Returns:
            (None)
        """
        if self.similarity_matrix is None:
            self.fit()

        similarity_matrix = self.similarity_matrix.astype('float')

        plt.figure()
        plt.imshow(similarity_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Similarity Matrix - {}".format(self.distance_measure))
        plt.colorbar()
        tick_mark = np.arange(len(self.classes_list))
        plt.xticks(tick_mark, self.classes_list, rotation=45)
        plt.yticks(tick_mark, self.classes_list)

        fmt = '.2f'
        thresh = similarity_matrix.max() / 2.
        for i, j in itertools.product(range(similarity_matrix.shape[0]), range(similarity_matrix.shape[1])):
            plt.text(j, i, format(similarity_matrix[i, j], fmt), horizontalalignment="center",
                     color="white" if similarity_matrix[i, j] > thresh else "black")

        plt.ylabel('Classes')
        plt.xlabel('Classes')
        if is_save:
            plt.savefig("BBox_Distribution_Similarity_Matrix-{}.png".format(self.distance_measure))
        plt.show()

    def report_modeling(self, is_save):
        """
        Display and save figure about each class box distribution and linear regression result

        Args:
            (bool) is_save : if True, save figure
                                False, not save figure

        Returns:
            (None)
        """

        float_subplotsize = np.sqrt(len(self.classes_list))
        floor_subplotsize = np.floor(float_subplotsize)

        if (float_subplotsize - floor_subplotsize) != 0:
            floor_subplotsize += 1

        subplotsize = [int(floor_subplotsize) for _ in range(2)]

        plt.figure()
        for idx, class_label in enumerate(self.classes_list):

            slope = self.analysis_result[class_label]['slope']
            bias = self.analysis_result[class_label]['bias']

            x = np.arange(0, 1, 0.1)
            y = list()
            for _x in x:
                y.append(bias + (slope * _x))

            plt.subplot(subplotsize[0], subplotsize[1], idx + 1)
            bbox_reshape = np.transpose(np.asarray(self.bbox_data[class_label]))
            plt.scatter(bbox_reshape[0], bbox_reshape[1], label=class_label)
            plt.plot(x, y, color='r')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.legend()
            plt.title(class_label)

        if is_save is True:
            plt.savefig("each_Classes.png")
        plt.show()

    @staticmethod
    def _distribution_modeling(classes_list, bbox_data) -> Dict:
        """

        Args:
            classes_list (List) : classes name
            bbox_data (Dict) : bbox information as each classes

        Returns:
            (Dict) : result of Linear Regression as following

                    {
                        "classes" : {
                                        "score": (List), coefficient of determination
                                        "slope": (List),
                                        "bias": (List),
                                        "vec": (List), (x, y) vector when x is 1.0
                            }
                        ...
                    }
        """

        analysis_result = dict()
        for classes in classes_list:
            analysis_result.update({classes: {}})

        for classes in classes_list:
            data = np.transpose(np.asarray(bbox_data[classes]))
            x = data[0].reshape(-1, 1)
            y = data[1]
            model = LinearRegression().fit(x, y)
            score = model.score(x, y)
            slope = model.coef_
            bias = model.intercept_
            vec_x = np.array([[1.0]])
            vec_y = model.predict(vec_x)

            analysis_result[classes] = {"score": score,
                                        "slope": slope[0],
                                        "bias": bias,
                                        "vec": [vec_x[0][0], vec_y[0]]}

        return analysis_result

    def _similarity_intra_classes(self, analysis_result: Dict) -> np.ndarray:
        """
        Calculate similarity intra classes distribution

        Args:
            analysis_result (Dict): result of Linear Regression as following

                    {
                        "classes" : {
                                        "score": (List), coefficient of determination
                                        "slope": (List),
                                        "bias": (List),
                                        "vec": (List), (x, y) vector when x is 1.0
                            }
                        ...
                    }

        Returns:
            (np.ndarray) : similarity matrix like a confusion matrix
        """

        similarity_matrix = []

        for source_classes in self.classes_list:
            source_vec = self._unit_vector(np.asarray(analysis_result[source_classes]["vec"]))

            rows = list()
            for target_classes in self.classes_list:
                target_vec = self._unit_vector(np.asarray(analysis_result[target_classes]["vec"]))
                similarity_score = self.metric(source_vec, target_vec)
                rows.append(similarity_score)
            similarity_matrix.append(rows)

        similarity_matrix = np.asarray(similarity_matrix)

        return similarity_matrix

    @staticmethod
    def _unit_vector(vector: np.ndarray) -> np.ndarray:
        return vector / np.linalg.norm(vector, 2)

    # Distance Measure
    @staticmethod
    def _inner_product(source_vector: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
        return np.inner(source_vector, target_vector)

    @staticmethod
    def _manhattan_distance(source_vector: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
        return np.abs(source_vector[0] - target_vector[0]) + np.abs(source_vector[1] - target_vector[1])

    @staticmethod
    def _euclidean_distance(source_vector: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
        return np.sqrt(
            np.power(source_vector[0] - target_vector[0], 2) +
            np.power(source_vector[1] - target_vector[1], 2))

    def _cosine_similarity(self, source_vector: np.ndarray, target_vector: np.ndarray) -> np.ndarray:
        return self._inner_product(source_vector, target_vector) / \
               np.linalg.norm(source_vector, 2) * np.linalg.norm(target_vector, 2)

    @staticmethod
    def _collect_classes(anno: DetectionAnnotations) -> List:
        objs = sum([FILE.OBJECTS for FILE in anno.FILES], [])
        classes_info = Counter([obj.CLASS for obj in objs])

        return list(classes_info.keys())

    def _collect_bbox(self, anno: DetectionAnnotations, classes_list: List) -> Dict:
        """
        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
            classes_list (List) : classes names

        Returns:
            (Dict) : each classes bbox distribution as follow
            {
                "(name of class)" : [[normalized bbox width (float), normalized bbox height], ...]
                ...
            }
        """

        class_bbox = dict()
        for class_label in classes_list:
            class_bbox.update({class_label: []})

        obj_files = [FILE for FILE in anno.FILES]
        for obj_file in obj_files:
            for obj in obj_file.OBJECTS:
                class_bbox[obj.CLASS].append(self._bbox_normalize(obj_file.IMAGE_WIDTH,
                                                                  obj_file.IMAGE_HEIGHT,
                                                                  obj.XMIN,
                                                                  obj.YMIN,
                                                                  obj.XMAX,
                                                                  obj.YMAX))
        return class_bbox

    @staticmethod
    def _bbox_normalize(image_width: int,
                        image_height: int,
                        xmin: int,
                        ymin: int,
                        xmax: int,
                        ymax: int) -> List:

        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        norm_w = bbox_w / image_width
        norm_h = bbox_h / image_height
        return [norm_w, norm_h]


class BBoxDimensionAnalyzer:

    def __init__(self,
                 anno: DetectionAnnotations,
                 num_cetroid: int = 5,
                 distance_measure: str = 'iou'):
        """
        BBox Dimension Clustering

        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
            num_cetroid : number of centroid for kmeans
            distance_measure : distance measure of kemans
        """
        classes_list = self._collect_classes(anno)
        self.anno = anno
        self.classes_list = classes_list
        self.bbox_data = self._collect_bbox(anno, classes_list)
        self.number_of_centroid = num_cetroid
        self.distance_measure = distance_measure
        self.kmeans_result = None

        # TODO should be implementation about calc similarity bbox distribution & merge

    @staticmethod
    def _collect_classes(anno: DetectionAnnotations) -> List:
        objs = sum([FILE.OBJECTS for FILE in anno.FILES], [])
        classes_info = Counter([obj.CLASS for obj in objs])

        return list(classes_info.keys())

    def _collect_bbox(self, anno: DetectionAnnotations, classes_list: List) -> Dict:
        """
        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
            classes_list (List) : classes names

        Returns:
            (Dict) : each classes bbox distribution as follow
            {
                "(name of class)" : [[normalized bbox width (float), normalized bbox height], ...]
                ...
            }
        """

        class_bbox = dict()
        for class_label in classes_list:
            class_bbox.update({class_label: []})

        obj_files = [FILE for FILE in anno.FILES]
        for obj_file in obj_files:
            for obj in obj_file.OBJECTS:
                class_bbox[obj.CLASS].append(self._bbox_normalize(obj_file.IMAGE_WIDTH,
                                                                  obj_file.IMAGE_HEIGHT,
                                                                  obj.XMIN,
                                                                  obj.YMIN,
                                                                  obj.XMAX,
                                                                  obj.YMAX))
        return class_bbox

    @staticmethod
    def _bbox_normalize(image_width: int,
                        image_height: int,
                        xmin: int,
                        ymin: int,
                        xmax: int,
                        ymax: int) -> List:

        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        norm_w = bbox_w / image_width
        norm_h = bbox_h / image_height
        return [norm_w, norm_h]

    def fit(self) -> Dict:
        """
        Find prior boxes using Kmeans

        Args:
            (None)
        Returns
            (Dict): dimension clustering result about each classes as follow
            {
                (str) "classes": {
                                    (str) "centroid" : (List) founded centroid coordinates
                                 },
                ...
            }

        """

        self.kmeans_result = dict()
        for classes in self.classes_list:
            self.kmeans_result.update({classes: {}})

        return_value = dict()
        for classes in self.classes_list:
            return_value.update({classes: {}})

        classes_list = self.classes_list

        for classes in classes_list:
            data = self.bbox_data[classes]

            centroid_candidate = kmeans_plusplus_initializer.FARTHEST_CENTER_CANDIDATE
            centroid_initializer = kmeans_plusplus_initializer(
                data=data,
                amount_centers=self.number_of_centroid,
                amount_candidates=centroid_candidate)
            init_centroid = centroid_initializer.initialize()

            if self.distance_measure == 'iou':
                metric = distance_metric(type_metric.USER_DEFINED, func=self._iou)
            elif self.distance_measure == 'l2':
                metric = distance_metric(type_metric.EUCLIDEAN)
            elif self.distance_measure == 'l1':
                metric = distance_metric(type_metric.MANHATTAN)
            else:
                raise RuntimeError("Not supported {} metric".format(self.distance_measure))

            k_means = kmeans(data=data,
                             initial_centers=init_centroid,
                             metric=metric)
            k_means.process()
            clusters = k_means.get_clusters()
            centers = k_means.get_centers()

            return_value[classes]['centroid'] = centers
            self.kmeans_result[classes]["init_centroid"] = init_centroid
            self.kmeans_result[classes]["clusters"] = clusters
            self.kmeans_result[classes]['centers'] = centers

        return return_value

    def report(self) -> None:
        """
        Visualization Kmeans result
        """
        if self.kmeans_result is None:
            raise RuntimeError("Member variable `result` or `classes_frequency` is None"
                               "it should be run `fit` function first")

        for classes in self.classes_list:
            visualizer = cluster_visualizer(["result of kmeans in class `{}`".format(classes)])
            visualizer.append_cluster(self.bbox_data[classes])
            visualizer.append_clusters(clusters=self.kmeans_result[classes]['clusters'],
                                       data=self.bbox_data[classes])
            visualizer.append_cluster(self.kmeans_result[classes]['centers'],
                                      marker='*',
                                      markersize=10)
            visualizer.show(invisible_axis=False)

    @staticmethod
    def _iou(point1, point2):
        """
        Calculate IOU
        """
        point1 = 100 * point1
        point2 = 100 * point2

        axis = 1 if len(point1.shape) > 1 else 0

        p1_area = np.prod(point1, axis=axis)
        p2_area = np.prod(point2, axis=axis)
        intersection = np.minimum(p1_area, p2_area)
        union = np.maximum(p1_area, p2_area)
        iou = intersection / union
        iou_distance = 1 - iou

        return iou_distance


class ClassDistributionAnalyzer:
    def __init__(self, anno: DetectionAnnotations) -> None:
        """
        Class Distribution Analyzer

        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
        """
        self.anno = anno
        self.classes_list = None
        self.classes_frequency = None

    def fit(self) -> Tuple[List, List]:
        """
        collect classes name in dataset and counter classes frequency

        Args:
            (None)

        Returns:
            (List, List) : name of classes, frequency of classes
        """
        objs = sum([FILE.OBJECTS for FILE in self.anno.FILES], [])
        classes_info = Counter([obj.CLASS for obj in objs])
        self.classes_list = classes_info.keys()
        self.classes_frequency = classes_info.values()
        return list(self.classes_list), list(self.classes_frequency)

    def report(self, is_save: bool = True) -> None:
        """
        Display and save figure about Class distribution
        Args:
            is_save (bool) : if True, save figure about class distribution
                                False, not save figure about class ddistribution
        Returns:
            (None)

        """
        if (self.classes_list is None) or (self.classes_frequency is None):
            raise RuntimeError("Member variable `classes_list` or `classes_frequency` is None"
                               "it should be run `fit` function first")

        print("List of Classes : {}".format(self.classes_list))
        print("List of Classes Frequency : {}".format(self.classes_frequency))

        plt.figure()
        plt.bar(self.classes_list, self.classes_frequency)
        plt.xlabel("Classes")
        plt.ylabel("Frequency")
        if is_save:
            plt.savefig("Class histogram.png")
        plt.show()


class BBoxAnalyzer:
    def __init__(self, anno: DetectionAnnotations):
        """
        BBox Analyzer

        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
        """
        self.anno = anno
        self.classes_list = None
        self.bbox_data = None

    def fit(self):
        self.classes_list = self._collect_classes()
        self.bbox_data = self._collect_bbox()

    def report(self, is_save: bool = True):
        """
        Display and save figure about bbox distribution

        Args:
            (bool) is_save : if True, save figure about bbox distribution
                                False, not save figure about bbox ddistribution

        Returns:
            (None)
        """
        if (self.bbox_data is None) or (self.classes_list is None):
            raise RuntimeError("Member variable `bbox_data` or `classes_list` is None"
                               "it should be run `fit` function first")
        self._report_all_classes(is_save=is_save)
        self._report_each_classes(is_save=is_save)

    def _report_all_classes(self, is_save):
        """
        Display and save figure about all class box distribution

        Args:
            (bool) is_save : if True, save figure about bbox distribution
                                False, not save figure about bbox ddistribution

        Returns:
            (None)
        """

        plt.figure()

        for class_label in self.classes_list:
            bbox_reshape = np.transpose(np.asarray(self.bbox_data[class_label]))
            plt.scatter(bbox_reshape[0], bbox_reshape[1], label=class_label)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.legend()

        if is_save is True:
            plt.savefig("Whole_Classes.png")
        plt.title("Whole_Classes")
        plt.show()

    def _report_each_classes(self, is_save):
        """
        Display and save figure about each class box distribution

        Args:
            (bool) is_save : if True, save figure about bbox distribution
                                False, not save figure about bbox ddistribution

        Returns:
            (None)
        """

        float_subplotsize = np.sqrt(len(self.classes_list))
        floor_subplotsize = np.floor(float_subplotsize)

        if (float_subplotsize - floor_subplotsize) != 0:
            floor_subplotsize += 1

        subplotsize = [int(floor_subplotsize) for _ in range(2)]
        plt.figure()
        for idx, class_label in enumerate(self.classes_list):
            plt.subplot(subplotsize[0], subplotsize[1], idx + 1)
            bbox_reshape = np.transpose(np.asarray(self.bbox_data[class_label]))
            plt.scatter(bbox_reshape[0], bbox_reshape[1], label=class_label)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.legend()
            plt.title(class_label)

        if is_save is True:
            plt.savefig("each_Classes.png")
        plt.show()

    def _collect_classes(self) -> List:
        objs = sum([FILE.OBJECTS for FILE in self.anno.FILES], [])
        classes_info = Counter([obj.CLASS for obj in objs])

        return list(classes_info.keys())

    def _collect_bbox(self) -> Dict:
        """
        Collect BBox information

        Args:
            (None)

        Returns:
            (Dict) : each classes bbox distribution as follow
            {
                "(name of class)" : [[normalized bbox width (float), normalized bbox height], ...]
                ...
            }
        """

        class_bbox = dict()
        for class_label in self.classes_list:
            class_bbox.update({class_label: []})

        obj_files = [FILE for FILE in self.anno.FILES]
        for obj_file in obj_files:
            for obj in obj_file.OBJECTS:
                class_bbox[obj.CLASS].append(self._bbox_normalize(obj_file.IMAGE_WIDTH,
                                                                  obj_file.IMAGE_HEIGHT,
                                                                  obj.XMIN,
                                                                  obj.YMIN,
                                                                  obj.XMAX,
                                                                  obj.YMAX))
        return class_bbox

    @staticmethod
    def _bbox_normalize(image_width: int,
                        image_height: int,
                        xmin: int,
                        ymin: int,
                        xmax: int,
                        ymax: int) -> List:

        bbox_w = xmax - xmin
        bbox_h = ymax - ymin
        norm_w = bbox_w / image_width
        norm_h = bbox_h / image_height
        return [norm_w, norm_h]


if __name__ == "__main__":
    normal_case = [
        {
            "filepath": "",
            "image_width": 1920,
            "image_height": 1080,
            "objects": [
                {
                    "class": "a",
                    "xmin": 394,
                    "ymin": 528,
                    "xmax": 466,
                    "ymax": 724
                },

                {
                    "class": "a",
                    "xmin": 528,
                    "ymin": 0,
                    "xmax": 618,
                    "ymax": 726
                },
                {
                    "class": "b",
                    "xmin": 43,
                    "ymin": 613,
                    "xmax": 177,
                    "ymax": 1069
                }
            ]
        },

        {
            "filepath": "",
            "image_width": 1920,
            "image_height": 1080,
            "objects": [
                {
                    "class": "c",
                    "xmin": 1199,
                    "ymin": 83,
                    "xmax": 1240,
                    "ymax": 588
                }
            ]
        }
    ]
    anno = DetectionAnnotations(normal_case)
    class_analyzer = ClassDistributionAnalyzer(anno=anno)
    class_analyzer.fit()
    class_analyzer.report(is_save=False)

    bbox_analyzer = BBoxAnalyzer(anno=anno)
    bbox_analyzer.fit()
    bbox_analyzer.report(is_save=False)

    kmeans_case = [
        {
            "filepath": "",
            "image_width": 1920,
            "image_height": 1080,
            "objects": [
                {
                    "class": "a",
                    "xmin": 394,
                    "ymin": 528,
                    "xmax": 466,
                    "ymax": 724
                },

                {
                    "class": "a",
                    "xmin": 528,
                    "ymin": 0,
                    "xmax": 618,
                    "ymax": 726
                },
                {
                    "class": "a",
                    "xmin": 43,
                    "ymin": 613,
                    "xmax": 177,
                    "ymax": 1069
                }
            ]
        },

        {
            "filepath": "",
            "image_width": 1920,
            "image_height": 1080,
            "objects": [
                {
                    "class": "a",
                    "xmin": 1199,
                    "ymin": 83,
                    "xmax": 1240,
                    "ymax": 588
                },
                {
                    "class": "a",
                    "xmin": 1069,
                    "ymin": 771,
                    "xmax": 1145,
                    "ymax": 993
                },
                {
                    "class": "a",
                    "xmin": 1419,
                    "ymin": 819,
                    "xmax": 1476,
                    "ymax": 989
                }
            ]
        }
    ]

    annotations = DetectionAnnotations(kmeans_case)
    dimension_cluster = BBoxDimensionAnalyzer(anno=annotations,
                                              num_cetroid=2,
                                              distance_measure='iou')
    prior_boxes = dimension_cluster.fit()
    dimension_cluster.fit()
    dimension_cluster.report()
