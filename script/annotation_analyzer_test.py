import csv
from tqdm import tqdm
from utils.annotation_analyzer import ClassDistributionAnalyzer, BBoxAnalyzer, BBoxDimensionAnalyzer
from utils.annotation_interfaces import DetectionAnnotations

CSV_FILEPATH = "sample_data.csv"

annotations = list()

file = open(CSV_FILEPATH)
numline = len(file.readlines())

with open(CSV_FILEPATH) as csvfile:
    csvreader = csv.reader(csvfile)

    for idx, row in enumerate(tqdm(csvreader, total=numline)):
        if idx == 0:
            continue
        filepath = str(row[0])
        img_width = int(row[1])
        img_height = int(row[2])
        class_label = str(row[3])
        xmin = int(row[4])
        ymin = int(row[5])
        xmax = int(row[6])
        ymax = int(row[7])

        annotations.append({"filepath": filepath,
                            "image_width": img_width,
                            "image_height": img_height,
                            "objects": [{
                                "class": class_label,
                                "xmin": xmin,
                                "ymin": ymin,
                                "xmax": xmax,
                                "ymax": ymax
                            }]})

detection_anno = DetectionAnnotations(annotations)

class_analyzer = ClassDistributionAnalyzer(anno=detection_anno)
class_analyzer.fit()
class_analyzer.report(is_save=False)

bbox_analyzer = BBoxAnalyzer(anno=detection_anno)
bbox_analyzer.fit()
bbox_analyzer.report(is_save=False)

bbox_dimension_analyzer = BBoxDimensionAnalyzer(anno=detection_anno,
                                                num_cetroid=5,
                                                distance_measure='iou')
prior_boxes = bbox_dimension_analyzer.fit()
print("Prior_boxes : {}".format(prior_boxes))
bbox_dimension_analyzer.report()
