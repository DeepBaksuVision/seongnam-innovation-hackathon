import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Tuple, Dict
from utils.detection_annotations import DetectionAnnotations, DetectionObject


class AnnotationAnalyzer:

    def __init__(self, anno: DetectionAnnotations):
        """
        Annotation Analyzer

        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
        """
        self.classes, self.class_frequency = self._collect_classes_and_frequency(anno)
        self.class_bbox_distrib = self._bbox_distribution(anno, self.classes)

    @staticmethod
    def _collect_classes_and_frequency(anno: DetectionAnnotations) -> Tuple[List, List]:
        """
        collect classes name in dataset and counter classes frequency

        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object

        Returns:
            (List, List) : name of classes, frequency of classes
        """
        objs = sum([FILE.OBJECTS for FILE in anno.FILES], [])
        classes_info = Counter([obj.CLASS for obj in objs])
        return classes_info.keys(), classes_info.values()

    def _bbox_distribution(self,
                           anno: DetectionAnnotations,
                           classes: List) -> Dict:
        """
        Analyze BBox distribution

        Args:
            anno (DetectionAnnotations) : DetectionAnnotations object
            classes (List) : name of classes

        Returns:
            (Dict) : each classes bbox distribution as follow
                    {
                        "(name of class)" : [[normalized bbox width (float), normalized bbox height], ...]
                        ...
                    }
        """

        class_bbox_distrib = dict()
        for cls in classes:
            class_bbox_distrib.update({cls:[]})

        obj_files = [FILE for FILE in anno.FILES]
        for obj_file in obj_files:
            for obj in obj_file.OBJECTS:
                class_bbox_distrib[obj.CLASS].append(self._bbox_normalize(obj_file.IMAGE_WIDTH,
                                                                          obj_file.IMAGE_HEIGHT,
                                                                          obj.XMIN,
                                                                          obj.YMIN,
                                                                          obj.XMAX,
                                                                          obj.YMAX))
        return class_bbox_distrib

    def _bbox_normalize(self,
                        image_width: int,
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

    def show_class_distribution(self, savefig: bool = True) -> None:
        plt.figure()
        plt.bar(self.classes, self.class_frequency)
        plt.xlabel("Classes")
        plt.ylabel("Frequency")
        if savefig:
            plt.savefig("Class histogram.png")
        plt.show()

    def show_bbox_distribution(self, integrated: bool = False, savefig: bool = True) -> None:
        """
        Display BBox distrubition and save figure

        Args:
            integrated (Bool) : if True, display bbox distribution about all classes
                                False, display bbox distribution about each classes

            savefig (Bool) : if True, save figure about bbox distribution
                            False, not save figure about bbox distribution
        Returns:
            (None)
        """

        if integrated:
            plt.figure()

        for cls in self.classes:
            distrib_reshape = np.transpose(np.asarray(self.class_bbox_distrib[cls]))

            if not integrated:
                plt.figure()

            plt.scatter(distrib_reshape[0], distrib_reshape[1], label=cls)
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.title(cls)

            if not integrated:
                if savefig:
                    plt.savefig("".join([cls, ".png"]))
                plt.legend()
                plt.show()

        if integrated:
            if savefig:
                plt.savefig("".join(["integrated", ".png"]))
            plt.legend()
            plt.title("Integrated")
            plt.show()


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

    annotations = DetectionAnnotations(normal_case)
    analzer = AnnotationAnalyzer(annotations)
    # ['a', 'b', 'c']
    print(analzer.classes)
    # [2, 1, 1]
    print(analzer.class_frequency)
    # {'a': [[0.0375, 0.1814814814814815], [0.046875, 0.6722222222222223]], 'b': [[0.06979166666666667, 0.4222222222222222]], 'c': [[0.021354166666666667, 0.4675925925925926]]}
    print(analzer.class_bbox_distrib)
    analzer.show_bbox_distribution(integrated=False, savefig=False)
    analzer.show_bbox_distribution(integrated=True, savefig=False)
