import csv
from tqdm import tqdm
from utils.bbox_distribution import AnnotationAnalyzer
from utils.detection_annotations import DetectionAnnotations

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
                                "xmin" : xmin,
                                "ymin" : ymin,
                                "xmax" : xmax,
                                "ymax" : ymax
                            }]})

detection_anno = DetectionAnnotations(annotations)
anno_analyzer = AnnotationAnalyzer(detection_anno)
anno_analyzer.show_class_distribution(is_save=False)
anno_analyzer.show_bbox_distribution(each_classe=True, is_save=False)
anno_analyzer.show_bbox_distribution(each_classe=False, is_save=False)

