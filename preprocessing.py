from csv import reader
import cv2
import numpy as np
from PIL import Image
import os
import pydicom as dicom
import PIL # optional
import pandas as pd
import matplotlib.pyplot as plt
import copy
import json

csv_path = "/home/aralab/Documents/Mammo/test1.csv"
raw_path = "/home/aralab/Documents/Mammo/"
output_path = "/home/aralab/Documents/test_process/"

def globalBinarise(img, thresh, maxval):

    binarised_img = np.zeros(img.shape, np.uint8)
    binarised_img[img >= thresh] = maxval

    return binarised_img

def editMask(mask, ksize=(23, 23), operation="open"):

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)

    if operation == "open":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Then dilate
    edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)

    return edited_mask

def sortContoursByArea(contours, reverse=True):

    # Sort contours based on contour area.
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)

    # Construct the list of corresponding bounding boxes.
    bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]

    return sorted_contours, bounding_boxes

def xLargestBlobs(mask, top_x=None, reverse=True):

    # Find all contours from binarised image.
    # Note: parts of the image that you want to get should be white.
    contours, hierarchy = cv2.findContours(
        image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )

    n_contours = len(contours)

    # Only get largest blob if there is at least 1 contour.
    if n_contours > 0:

        # Make sure that the number of contours to keep is at most equal
        # to the number of contours present in the mask.
        if n_contours < top_x or top_x == None:
            top_x = n_contours

        # Sort contours based on contour area.
        sorted_contours, bounding_boxes = sortContoursByArea(
            contours=contours, reverse=reverse
        )

        # Get the top X largest contours.
        X_largest_contours = sorted_contours[0:top_x]

        # Create black canvas to draw contours on.
        to_draw_on = np.zeros(mask.shape, np.uint8)

        # Draw contours in X_largest_contours.
        X_largest_blobs = cv2.drawContours(
            image=to_draw_on,  # Draw the contours on `to_draw_on`.
            contours=X_largest_contours,  # List of contours to draw.
            contourIdx=-1,  # Draw all contours in `contours`.
            color=1,  # Draw the contours in white.
            thickness=-1,  # Thickness of the contour lines.
        )

    return n_contours, X_largest_blobs

def applyMask(img, mask):

    masked_img = img.copy()
    masked_img[mask == 0] = 0

    return masked_img

json_categories = '{ "categories": [{"id": 0,"name": "nothing"},{"id": 1,"name": "mass"}],'
json_image = ' "images": ['
json_annotations = ' "annotations": ['

cate = 0
i = 0
with open(csv_path, "r") as f:
    csv_reader = reader(f)
    header = next(csv_reader)
    # Check file as empty
    if header != None:
        # Iterate over each row after the header in the csv
        # print(type(csv_reader))
        for row in csv_reader:
            bbox_coordinates = '[]'
            # i += 1
            # extract folder and file name
            folder_name = row[0]
            file_name = row[2]
            laterality = row[3]
            #print(os.path.join(raw_path, folder_name, file_name + '.dicom'))
            # open dicom file
            ds = dicom.dcmread(os.path.join(raw_path, folder_name, row[2] + '.dicom'))
            pixel_array_numpy = ds.pixel_array
            # plt.imshow(pixel_array_numpy)
            # plt.show()
            # make folder
            if not (os.path.exists(output_path + folder_name)):
                os.makedirs(output_path + folder_name)
                # Normalized
            pixel_array_numpy = pixel_array_numpy.astype('float64')
            pixel_array_numpy *= 255 / pixel_array_numpy.max()
            # Binarized
            binarised_img = globalBinarise(img=pixel_array_numpy, thresh=10, maxval=255)
            #
            edited_mask = editMask(
                mask=binarised_img
            )
            #
            _, xlargest_mask = xLargestBlobs(mask=edited_mask, top_x=1)
            #
            masked_img = applyMask(img=pixel_array_numpy, mask=xlargest_mask)
            # cropped based on size
            a = (masked_img > 0).nonzero()
            a = np.array(a)[0:2, :]
            a = a.T
            min_y = np.min(a[:, 0])
            max_y = np.max(a[:, 0])
            min_x = np.min(a[:, 1])
            max_x = np.max(a[:, 1])
            cropped_img = copy.deepcopy(masked_img)
            cropped_img = cropped_img[min_y:max_y, min_x:max_x]
            # FLIP if R
            if laterality == "R":
                cropped_img = cv2.flip(cropped_img,1)
            # identify category
            if row[9] == "['No Finding']":
                cate = 0
            else:
                cate = 1
                # bounding box coordinates editing with cropped images
                bbox_coordinates = "[{},{},{},{}]".format(int(float(row[11]) - min_x), int(float(row[12]) - min_y),
                                                          int(float(row[13]) - min_x), int(float(row[14]) - min_y))
            # if row[9] !="['No Finding']":
            #     cropped_img = cv2.rectangle(cropped_img.astype('uint8'), (int(float(row[11])) - min_x, int(float(row[12])) - min_y),
            #                                 (int(float(row[13])) - min_x, int(float(row[14])) - min_y), (255, 0, 0), 10)
            # plt.imshow(cropped_img)
            # plt.show()
            new_file_name = "{}_{}_{}.png".format(file_name, laterality, row[4])
            cv2.imwrite(os.path.join(output_path, folder_name, new_file_name) , cropped_img)
            (height, width) = cropped_img.shape
            #print(height)
            # apparently, image_id can't have letter and numbers so...... gotta change each alphabet to its corresponding number
            for letter in file_name:
                if not letter.isdigit():
                    file_name = file_name.replace(str(letter), str(ord(letter)))
            # also can't have 0 as the beginning
            if file_name[0] == "0":
                file_name = "1" + file_name
            json_image += '{"file_name":"'+new_file_name + '", "height": ' + str(height) + ', "width": '+ str(width) +', "id": '+file_name+'},'
            json_annotations += '{"image_id": '+file_name+', "bbox": ' + bbox_coordinates + ', "category_id": ' + str(cate) + '},'
json_image = json_image[:-1]
json_image += '],'
json_annotations = json_annotations[:-1]
json_annotations += '] }'
# print(json_image)
# print(json_annotations)
# print(i)
json_everything = json_categories + json_image + json_annotations
print(json_everything)
json_object = json.loads(json_everything)
# Writing to sample.json
with open("sample.json", "w") as outfile:
    outfile.write(json_everything)