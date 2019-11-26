from typing import List, Dict

"""
Object Detection Annotations Interface
[
    {
        "filepath": (str),
        "image_width" : (int),
        "image_height": (int),
        "objects":
                    [
                        {
                            "class" : (str),
                            "xmin" : (int),
                            "ymin" : (int),
                            "xmax" : (int),
                            "ymax" : (int)
                        },
                        ...
                    ]

    }
    ...
]
"""


class _BaseAnnoComponents:

    def __init__(self, info, required_keys=[]):
        self.REQUIRED_KEYS = required_keys

        self._validate_keys(info)

    def _validate_keys(self, info):
        info_keys = info.keys()
        validate_elements = [KEY in info_keys for KEY in self.REQUIRED_KEYS]
        if len(validate_elements) != sum(validate_elements):
            indices = [i for i, is_valid in enumerate(validate_elements) if is_valid is False]
            missing_keys = [self.REQUIRED_KEYS[idx] for idx in indices]
            raise RuntimeError("Must contain all required key, "
                               "missing key are `{}`".format(",".join(missing_keys)))

    def dump(self):
        raise NotImplementedError


class DetectionObject(_BaseAnnoComponents):

    def __init__(self, object_info: Dict) -> None:
        """
        Interface of Detection object property

        Args:
            object_info : object information as follow
                          {
                            "class" : (str), # class name
                            "xmin" : (int), # x coordinates of left top point
                            "ymin" : (int), # y coordinates of left top point
                            "xmax" : (int), # x coordinates of right bottom point
                            "ymax" : (int), # y coordinates of right bottom point
                          }
        """
        self.REQUIRED_KEYS = ["class", "xmin", "ymin", "xmax", "ymax"]
        super(DetectionObject, self).__init__(object_info, self.REQUIRED_KEYS)

        self.CLASS: str = str(object_info["class"])
        self.XMIN: int = int(object_info["xmin"])
        self.YMIN: int = int(object_info["ymin"])
        self.XMAX: int = int(object_info["xmax"])
        self.YMAX: int = int(object_info["ymax"])

    def dump(self):
        print("\t\tclass:\t{}".format(self.CLASS))
        print("\t\txmin:\t{}".format(self.XMIN))
        print("\t\tymin:\t{}".format(self.YMIN))
        print("\t\txmax:\t{}".format(self.XMAX))
        print("\t\tymax:\t{}\n".format(self.YMAX))


class DetectionFile(_BaseAnnoComponents):

    def __init__(self, files_info: Dict) -> None:
        """
        Interface of Detection Image property

        Args:
            files_info (Dict) : files information as contain dict as follow

                             {
                                "filepath": (str),
                                "image_width" : (int),
                                "image_height": (int),
                                "objects":
                                [
                                    {"class" : "",
                                     "xmin" : "",
                                     "ymin" : "",
                                     "xmax" : "",
                                     "ymax" : ""}
                                     ...
                                ]

                            }
        """
        self.REQUIRED_KEYS = ["filepath", "image_width", "image_height", "objects"]
        super(DetectionFile, self).__init__(files_info, self.REQUIRED_KEYS)

        self.FILEPATH: str = str(files_info["filepath"])
        self.IMAGE_WIDTH: int = int(files_info["image_width"])
        self.IMAGE_HEIGHT: int = int(files_info["image_height"])
        self.NUMBER_OF_OBJECTS: int = len(files_info["objects"])
        self.OBJECTS: List = self._parse_objects(files_info["objects"])

    @staticmethod
    def _parse_objects(objects: List) -> List:
        """
        parse object information from objects parameter
        """

        return [DetectionObject(obj) for obj in objects]

    def dump(self):
        print("\tfilepath:\t{}".format(self.FILEPATH))
        print("\timage width:\t{}".format(self.IMAGE_WIDTH))
        print("\timage height:\t{}".format(self.IMAGE_HEIGHT))
        print("\tnumber of objects:\t{}".format(self.NUMBER_OF_OBJECTS))
        [OBJ.dump() for OBJ in self.OBJECTS]


class DetectionAnnotations(_BaseAnnoComponents):

    def __init__(self, anno_info: List):
        """
        Interface of Detection annotation property

        Args:
            anno_info: detection annotation info as follow
                       [{
                            "filepath": (str),
                            "image_width" : (int),
                            "image_height": (int),
                            "objects":
                            [
                                {"class" : "",
                                 "xmin" : "",
                                 "ymin" : "",
                                 "xmax" : "",
                                 "ymax" : ""}
                                 ...
                            ]

                        }
                        ...
                        ]
        """

        self.NUMBER_OF_FILES = len(anno_info)
        self.FILES = self._parse_files(anno_info)

    def dump(self):
        print("\tnumber of files:\t{}".format(self.NUMBER_OF_FILES))
        [FILE.dump() for FILE in self.FILES]

    @staticmethod
    def _parse_files(anno_info: List):
        """
        parse annotation information from anno_info parameter
        """
        return [DetectionFile(anno) for anno in anno_info]


if __name__ == "__main__":
    # normal case
    case1 = [{"filepath": "",
              "image_width": 0,
              "image_height": 0,
              "objects":[
                            {
                                "class": "",
                                "xmin": 0,
                                "ymin": 0,
                                "xmax": 0,
                                "ymax": 0
                            },

                            {
                                "class": "",
                                "xmin": 0,
                                "ymin": 0,
                                "xmax": 0,
                                "ymax": 0
                            },

                            {
                                "class": "",
                                "xmin": 0,
                                "ymin": 0,
                                "xmax": 0,
                                "ymax": 0
                            }
                        ]
                },

                {
                    "filepath": "",
                    "image_width": 0,
                    "image_height": 0,
                    "objects":
                        [
                            {
                                "class": "",
                                "xmin": 0,
                                "ymin": 0,
                                "xmax": 0,
                                "ymax": 0
                            }
                        ]
                }
            ]

    # abnormal case2. missing filepath
    case2 = [
                {
                    "image_width": 0,
                    "image_height": 0,
                    "objects": [
                                    {
                                        "class": "",
                                        "xmin": 0,
                                        "ymin": 0,
                                        "xmax": 0,
                                        "ymax": 0
                                    },
                                    {
                                        "class": "",
                                        "xmin": 0,
                                        "ymin": 0,
                                        "xmax": 0,
                                        "ymax": 0
                                    },
                                    {
                                        "class": "",
                                        "xmin": 0,
                                        "ymin": 0,
                                        "xmax": 0,
                                        "ymax": 0
                                    }
                                ]
                },
                {
                    "filepath": "",
                    "image_width": 0,
                    "image_height": 0,
                    "objects":
                        [
                            {
                                "class": "",
                                "xmin": 0,
                                "ymin": 0,
                                "xmax": 0,
                                "ymax": 0
                            }
                        ]
                }
    ]

    # abnormal case3 missing `class`
    case3 = [
                {
                    "image_width": 0,
                    "image_height": 0,
                    "objects": [
                                    {
                                        "xmin": 0,
                                        "ymin": 0,
                                        "xmax": 0,
                                        "ymax": 0
                                    },
                                    {
                                        "class": "",
                                        "xmin": 0,
                                        "ymin": 0,
                                        "xmax": 0,
                                        "ymax": 0

                                    },
                                    {
                                        "class": "",
                                        "xmin": 0,
                                        "ymin": 0,
                                        "xmax": 0,
                                        "ymax": 0
                                    }
                                ]
                },
                {
                    "filepath": "",
                    "image_width": 0,
                    "image_height": 0,
                    "objects": [
                                    {
                                        "class": "",
                                        "xmin": 0,
                                        "ymin": 0,
                                        "xmax": 0,
                                        "ymax": 0
                                    }
                                ]
                }
            ]

    # abnormal case4 missing whole annotations
    case4 = []
    try:
        annotations = DetectionAnnotations(case1)
        annotations.dump()
    except Exception as e:
        print(e)

    try:
        annotations = DetectionAnnotations(case2)
        annotations.dump()
    except Exception as e:
        print(e)

    try:
        annotations = DetectionAnnotations(case3)
        annotations.dump()
    except Exception as e:
        print(e)

    try:
        annotations = DetectionAnnotations(case4)
        annotations.dump()
    except Exception as e:
        print(e)
