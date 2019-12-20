from typing import List
from utils.annotation_interfaces import DetectionAnnotations


class ClassClipper:

    def __init__(self, anno: DetectionAnnotations, white_list: List):
        self.anno = anno
        self.white_list = white_list

    def cut(self) -> DetectionAnnotations:
        for file_idx, file in enumerate(self.anno.FILES):
            for obj_idx, obj in enumerate(file.OBJECTS):
                if obj.CLASS not in self.white_list:
                    file.OBJECTS[obj_idx] = None

            file.OBJECTS = [obj for obj in file.OBJECTS if obj is not None]
            file.NUMBER_OF_OBJECTS = len(file.OBJECTS)

            if file.NUMBER_OF_OBJECTS == 0:
                self.anno.FILES[file_idx] = None

        self.anno.FILES = [file for file in self.anno.FILES if file is not None]
        self.anno.NUMBER_OF_FILES = len(self.anno.FILES)

        return self.anno


if __name__ == "__main__":

    testcase = [
        {
            "filepath": "1",
            "image_width": 0,
            "image_height": 0,
            "objects": [
                {
                    "class": "a",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0
                },

                {
                    "class": "a",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0
                },

                {
                    "class": "b",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0
                }
            ]
        },

        {
            "filepath": "2",
            "image_width": 0,
            "image_height": 0,
            "objects":
                [
                    {
                        "class": "b",
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": 0,
                        "ymax": 0
                    }
                ]
        },
        {
            "filepath": "3",
            "image_width": 0,
            "image_height": 0,
            "objects":
                [
                    {
                        "class": "c",
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": 0,
                        "ymax": 0
                    }
                ]
        },
        {
            "filepath": "4",
            "image_width": 0,
            "image_height": 0,
            "objects":
                [
                    {
                        "class": "a",
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": 0,
                        "ymax": 0
                    }
                ]
        },
        {
            "filepath": "5",
            "image_width": 0,
            "image_height": 0,
            "objects":
                [
                    {
                        "class": "a",
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": 0,
                        "ymax": 0
                    },
                    {
                        "class": "c",
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": 0,
                        "ymax": 0
                    }
                ]
        },
        {
            "filepath": "6",
            "image_width": 0,
            "image_height": 0,
            "objects":
                [
                    {
                        "class": "a",
                        "xmin": 0,
                        "ymin": 0,
                        "xmax": 0,
                        "ymax": 0
                    }
                ]
        }

    ]
    try:
        print("==================== Before ====================")
        annotations = DetectionAnnotations(testcase)
        annotations.dump()
        print("==================== After ====================")
        class_clipper = ClassClipper(annotations, white_list=["a"])
        annotations = class_clipper.cut()
        annotations.dump()
    except Exception as e:
        print(e)
