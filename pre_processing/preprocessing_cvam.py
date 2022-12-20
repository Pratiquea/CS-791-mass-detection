from csv import reader
import cv2
import numpy as np
from PIL import Image
import os
import pydicom as dicom
import PIL  # optional
import pandas as pd
import matplotlib.pyplot as plt
import copy
import json
import argparse
import subprocess
from scipy.signal import argrelextrema, find_peaks
import copy
from django.utils.crypto import get_random_string
import sys

# Run command
# python3 preprocessing.py -d ~/Downloads/Dataset/physionet.org/files/vindr-mammo/1.0.0/images/ -o ~/Downloads/Dataset/physionet.org/files/vindr-mammo/1.0.0/processed_imgs/
# import matplotlib
# print(matplotlib.get_backend())
class PreProcessing:
    def __init__(self, args, plot_imgs=False):
        self.csv_path = args.csv_path
        self.dataset_path = args.dataset_path
        self.output_path = args.output_path
        self.missing_img_txt = args.missing_img_txt
        self.processed_img_txt = args.processed_img_txt
        self.dataset_type = args.dataset_type
        self.processing_issue_img_txt = args.processing_issue_img_txt
        self.plot = plot_imgs
        self.all_ids = []
        self.mass_folder_train = []
        self.mass_folder_test = []
        self.mass_folder_val = []
        self.val_split = 0.2

    def gen_unique_id(self):
        u_id = get_random_string(8, allowed_chars='123456789')
        u_id = int(u_id)
        if u_id in self.all_ids:
            self.gen_unique_id()
        else:
            self.all_ids.append(u_id)
            return u_id

    def globalBinarise(self, img, thresh, maxval):

        binarised_img = np.zeros(img.shape, np.uint8)
        binarised_img[img >= thresh] = maxval

        return binarised_img

    def editMask(self, mask, ksize=(23, 23), operation="open"):

        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)

        if operation == "open":
            edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Then dilate
        edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)

        return edited_mask

    def sortContoursByArea(self, contours, reverse=True):

        # Sort contours based on contour area.
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)

        # Construct the list of corresponding bounding boxes.
        bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]

        return sorted_contours, bounding_boxes

    def xLargestBlobs(self, mask, top_x=None, reverse=True):

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
            sorted_contours, bounding_boxes = self.sortContoursByArea(
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

    def applyMask(self, img, mask):

        masked_img = img.copy()
        masked_img[mask == 0] = 0

        return masked_img

    def get_minima(self, img, plot=False):
        xlargest_mask = copy.deepcopy(img)
        distribution = np.count_nonzero(xlargest_mask >= 1, axis=1)
        distribution_for_visualisation = copy.deepcopy(distribution)
        row_axis = np.arange(0, distribution.shape[0])
        # maxima = argrelextrema(distribution, np.greater)
        maxima = np.argmax(distribution)
        # print("maxima = {}".format(maxima))
        maxima_val = distribution[maxima]
        distribution[distribution >= 0.7 * maxima_val] = maxima_val
        # hoping that maxima is found at the nipple region, hence exaggerate the minima to go beyond it.
        maxima += 500
        distribution_after_maxima = distribution[maxima:]
        row_axis_after_maxima = np.arange(0, len(distribution_after_maxima))

        minima = find_peaks(x=-distribution_after_maxima)
        # print("minima = {}".format(minima))

        crop_after_ind = None
        if len(minima[0]) > 0:
            if plot:
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.suptitle('Horizontally stacked subplots')
                # plt.plot(distribution, row_axis)
                ax1.plot(distribution_after_maxima, row_axis_after_maxima)
                ax2.plot(distribution_for_visualisation, row_axis)
                ax1.invert_yaxis()
                ax2.invert_yaxis()
            for each in minima:
                if type(each) is dict:
                    continue
                if type(each) is np.ndarray:
                    each = each[0]
                minima_list = np.ones((maxima,)) * (each)
                crop_after_ind = each + maxima
                minima_list_2 = np.ones((maxima,)) * (each + maxima)
                row_minima = np.arange(0, len(minima_list))
                if plot:
                    ax1.plot(row_minima, minima_list)
                    ax2.plot(row_minima, minima_list_2)
            if plot:
                plt.draw()
                plt.show()

        return crop_after_ind

    def run(self):
        print("\n")

        # json_categories_training = '{ "categories": [{"id": 0,"name": "mass"}],'
        # json_image_training = ' "images": ['
        # json_annotations_training = ' "annotations": ['
        json_categories_training = { "categories": [{"id": 0,"name": "mass"}]}
        json_image_training = []
        json_annotations_training = []

        json_categories_val = { "categories": [{"id": 0,"name": "mass"}]}
        json_image_val = []
        json_annotations_val = []

        json_categories_test = { "categories": [{"id": 0,"name": "mass"}]}
        json_image_test = []
        json_annotations_test = []

        cmd = ["df", "/", "-h"]

        cate = 0
        i = 0
        count = 0

        with open(self.csv_path, "r") as f:
            csv_reader = reader(f)
            header = next(csv_reader)
            if header != None:
                for row in csv_reader:
                    folder = row[0]
                    mass = row[9]
                    type = row[-1]
                    if type == 'training':
                        if 'mass' in mass.lower() and folder not in self.mass_folder_train:
                            self.mass_folder_train.append(folder)
                    elif type == 'test':
                        if 'mass' in mass.lower() and folder not in self.mass_folder_test:
                            self.mass_folder_test.append(folder)

        size = len(self.mass_folder_train)
        i = 0
        rand = np.random.randint(0,size -1, size= (round(size * self.val_split) ,))
        for ind_ in rand:
            self.mass_folder_val.append(self.mass_folder_train[ind_])
        for each in self.mass_folder_val:
            if each in self.mass_folder_train:
                self.mass_folder_train.remove(each)
        # common = []
        # for each in self.mass_folder_train:
        #     if each in self.mass_folder_val:
        #         common.append(each)
        self.mass_folder_train = set(self.mass_folder_train)
        self.mass_folder_test = set(self.mass_folder_test)
        self.mass_folder_val = set(self.mass_folder_val)

        for folder in self.mass_folder_train:
            os.makedirs(os.path.join(self.output_path, "training", folder))

        for folder in self.mass_folder_val:
            os.makedirs(os.path.join(self.output_path, "val", folder))

        for folder in self.mass_folder_test:
            os.makedirs(os.path.join(self.output_path, "test", folder))

        f.close()
        # sys.exit(0)
        with open(self.csv_path, "r") as f, \
                open(self.missing_img_txt, "w") as f2, \
                open(self.processed_img_txt, "w") as f3, \
                open(self.processing_issue_img_txt, "w") as f4:
            csv_reader = reader(f)
            header = next(csv_reader)
            # Check file as empty
            if header != None:
                # Iterate over each row after the header in the csv
                # print(type(csv_reader))
                for row in csv_reader:

                    # if count > 10:
                    #     break
                    # count += 1
                    # Check if there's enough space on the disk
                    a = subprocess.Popen(cmd, stdout=subprocess.PIPE)
                    space_left = a.communicate()[0].decode('utf-8').split()[10]
                    space_left = space_left.replace("G", "")
                    if float(space_left) < 1.0:
                        print("Not enough space left. exiting...")
                        break


                    # if dataset_type_ != self.dataset_type:
                    #     print(
                    #         "This image belongs to {} set. Skipping it since we are processing images for {} set".format(
                    #             dataset_type_, self.dataset_type))
                    #     continue
                    folder_name = row[0]

                    if folder_name in self.mass_folder_test or folder_name in self.mass_folder_train or folder_name in self.mass_folder_val:
                        dataset_type_ = row[-1]
                        if folder_name in self.mass_folder_val:
                            dataset_type_ = 'val'

                        defect_type = copy.deepcopy(row[9])
                        defect_type = defect_type.replace("[", "").replace("]", "").replace("'", "")
                        defect_type = defect_type.lower()
                        defect_type = defect_type.split(", ")
                        # print("\ndefect type = {}".format(defect_type))

                        # Only process images with mass
                        # if "mass" not in defect_type:
                        #     continue

                        bbox_coordinates = []
                        bbox_orig = []
                        bbox_csv_format = []

                        # i += 1
                        # extract folder and file name

                        file_name = row[2]
                        laterality = row[3]
                        # print(os.path.join(self.dataset_path, folder_name, file_name + '.dicom'))
                        # open dicom file
                        file_path = os.path.join(self.dataset_path, folder_name, row[2] + '.dicom')
                        # print("file_path = {}".format(file_path))
                        if not os.path.exists(file_path):
                            f2.write(file_path + '\n')
                            print("file_path = {} does not exist".format(file_path))
                            continue
                        else:
                            f3.write(file_path + '\n')
                            print("processing image = {}".format(file_path))
                        ds = dicom.dcmread(file_path)
                        pixel_array_numpy = ds.pixel_array
                        img_orig = copy.deepcopy(pixel_array_numpy)
                        max_img_y, max_img_x = pixel_array_numpy.shape
                        # print("max img x/width = {}, max img y/height = {}".format(max_img_x, max_img_y))
                        # plt.imshow(pixel_array_numpy)
                        # plt.show()
                        # make folder
                        output_path_ = os.path.join(self.output_path, dataset_type_, folder_name)
                        if not (os.path.exists(output_path_)):
                            os.makedirs(output_path_)
                            # Normalized
                        pixel_array_numpy = pixel_array_numpy.astype('float64')
                        pixel_array_numpy *= 255 / pixel_array_numpy.max()
                        # Binarized
                        binarised_img = self.globalBinarise(img=pixel_array_numpy, thresh=10, maxval=255)
                        #
                        edited_mask = self.editMask(
                            mask=binarised_img
                        )
                        #
                        _, xlargest_mask = self.xLargestBlobs(mask=edited_mask, top_x=1)
                        #

                        masked_img = self.applyMask(img=pixel_array_numpy, mask=xlargest_mask)

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
                            cropped_img = cv2.flip(cropped_img, 1)
                            img_orig_flip = cv2.flip(img_orig.copy(), 1)
                            # flip bounding box as well

                        # identify category
                        if row[9] == "['No Finding']":
                            cate = 0
                        else:
                            # category should be zero since we have only one class
                            cate = 0
                            # bounding box coordinates editing with cropped images
                            # bbox_coordinates = "[{},{},{},{}]".format(int(float(row[11]) - min_x), int(float(row[12]) - min_y),
                            #                                           int(float(row[13]) - min_x), int(float(row[14]) - min_y))
                            if laterality == "R":
                                mid_x = float(max_img_x) / 2
                                mid_y = float(max_img_y) / 2

                                bbox_orig.append(float(row[11]))
                                bbox_orig.append(float(row[12]))
                                bbox_orig.append(float(row[13]))
                                bbox_orig.append(float(row[14]))
                                bbox_orig_flip = [0, 0, 0, 0]
                                bbox_coordinates = [0, 0, 0, 0]
                                # print("mid_x = {} dist from mid x = {}".format(mid_x,(bbox_orig[0] - mid_x)))
                                bbox_orig_flip[0] = int(mid_x - (bbox_orig[0] - mid_x))
                                bbox_orig_flip[1] = int(bbox_orig[1])
                                bbox_orig_flip[2] = int(mid_x - (bbox_orig[2] - mid_x))
                                bbox_orig_flip[3] = int(bbox_orig[3])

                                bbox_coordinates[0] = int(bbox_orig_flip[0])
                                bbox_coordinates[1] = int(bbox_orig_flip[1] - min_y)
                                bbox_coordinates[2] = int(bbox_orig_flip[2])
                                bbox_coordinates[3] = int(bbox_orig_flip[3] - min_y)

                                # convert all bounding box to x,y,width,height format
                                bbox_orig[2] = bbox_orig[2] - bbox_orig[0]
                                bbox_orig[3] = bbox_orig[3] - bbox_orig[1]

                                bbox_orig_flip[2] = bbox_orig_flip[2] - bbox_orig_flip[0]
                                bbox_orig_flip[3] = bbox_orig_flip[3] - bbox_orig_flip[1]

                                bbox_coordinates[2] = bbox_coordinates[2] - bbox_coordinates[0]
                                bbox_coordinates[3] = bbox_coordinates[3] - bbox_coordinates[1]

                            else:
                                bbox_coordinates.append(int(float(row[11])))
                                bbox_coordinates.append(int(float(row[12]) - min_y))
                                bbox_coordinates.append(int(float(row[13])) - bbox_coordinates[0])
                                bbox_coordinates.append(int(float(row[14]) - min_y) - bbox_coordinates[1])

                                bbox_orig.append(int(float(row[11])))
                                bbox_orig.append(int(float(row[12])))
                                bbox_orig.append(int(float(row[13]) - float(row[11])))
                                bbox_orig.append(int(float(row[14]) - float(row[12])))

                                bbox_csv_format.append(int(float(row[11])))
                                bbox_csv_format.append(int(float(row[12])))
                                bbox_csv_format.append(int(float(row[13])))
                                bbox_csv_format.append(int(float(row[14])))

                        binarised_img = self.globalBinarise(img=cropped_img, thresh=10, maxval=255)
                        #
                        edited_mask = self.editMask(
                            mask=binarised_img
                        )

                        crop_after_ind = self.get_minima(edited_mask, False)

                        if crop_after_ind is not None and len(bbox_coordinates)>0:
                            cropped_img = cropped_img[:crop_after_ind, :]
                            # if crop is before y coord of bbox, then flag image
                            if crop_after_ind < bbox_coordinates[3]:
                                f4.write(file_path + '\n')
                                print("\n\n########### image processing failed for image = {}".format(file_path))
                                continue

                        plt_cropped_img = copy.deepcopy(cropped_img)
                        # print("bbox = {}".format(bbox_coordinates[0]))
                        # print("bbox = {}".format(bbox_coordinates[1]))
                        # print("bbox = {}".format(bbox_coordinates[2]))
                        # print("bbox = {}".format(bbox_coordinates[3]))

                        if self.plot:
                            fig, ax1, ax2, ax3 = None, None, None, None
                            if laterality == "R":
                                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                            else:
                                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                                # fig, (ax1, ax2) = plt.subplots(1,2)

                            # Maximize display window
                            mng = plt.get_current_fig_manager()
                            varrr = (int(mng.window.maxsize()[0] * 0.8), int(mng.window.maxsize()[1] * 0.8))
                            # print("type of mng = {}".format(type(varrr)))
                            # print("mng = {}".format(varrr))
                            mng.resize(*varrr)

                            plt_cropped_img = cv2.rectangle(plt_cropped_img.astype('uint8'),
                                                            (int(bbox_coordinates[0]), int(bbox_coordinates[1])), (
                                                            int(bbox_coordinates[2] + bbox_coordinates[0]),
                                                            int(bbox_coordinates[3] + bbox_coordinates[1])), (255, 0, 0),
                                                            10)
                            img_orig_copy = copy.deepcopy(img_orig.astype('uint8'))
                            img_orig = cv2.rectangle(img_orig.astype('uint8'), (int(bbox_orig[0]), int(bbox_orig[1])),
                                                     (int(bbox_orig[2] + bbox_orig[0]), int(bbox_orig[3] + bbox_orig[1])),
                                                     (255, 0, 0), 10)

                            ax1.imshow(plt_cropped_img, cmap='gray')
                            ax2.imshow(img_orig, cmap='gray')

                            if laterality == "R":
                                img_orig_flip = cv2.rectangle(img_orig_flip.astype('uint8'),
                                                              (int(bbox_orig_flip[0]), int(bbox_orig_flip[1])), (
                                                              int(bbox_orig_flip[2] + bbox_orig_flip[0]),
                                                              int(bbox_orig_flip[3] + bbox_orig_flip[1])), (255, 0, 0), 10)
                                ax3.imshow(img_orig_flip, cmap='gray')
                            else:
                                img_orig_csv = cv2.rectangle(img_orig_copy,
                                                             (int(bbox_csv_format[0]), int(bbox_csv_format[1])),
                                                             (int(bbox_csv_format[2]), int(bbox_csv_format[3])),
                                                             (255, 0, 0), 10)
                                ax3.imshow(img_orig_csv, cmap='gray')

                            # plt.show(block=False)
                            plt.show()
                            # plt.pause(2.0)
                            # plt.close()

                        print("init file name = {}".format(file_name))
                        # new_file_name = "{}_{}_{}.png".format(copy.deepcopy(file_name), laterality, row[4])
                        new_file_name = copy.deepcopy(file_name) + '_' + laterality + '_' + row[4] + '.png'
                        new_file_path = os.path.join(output_path_, new_file_name)
                        print("saving image to= {}".format(new_file_path))
                        new_file_name = os.path.join(folder_name, new_file_name)
                        # print("file_name for coco annotiation = {}".format(new_file_name))
                        cv2.imwrite(new_file_path, cropped_img)
                        (height, width) = cropped_img.shape
                        # resized = cv2.resize(cropped_img, (int(width/4), int(height/4)))
                        # cv2.imshow("processed image", resized)
                        # cv2.waitKey(0)
                        # print(height)
                        # apparently, image_id can't have letter and numbers so...... gotta change each alphabet to its corresponding number
                        for letter in folder_name:
                            if not letter.isdigit():
                                folder_name = folder_name.replace(str(letter), str(ord(letter)))
                        unique_img_id = self.gen_unique_id()
                        unique_id = self.gen_unique_id()
                        print("unique_img_id = {}".format(unique_img_id))
                        print("\n")
                        if dataset_type_ == 'training':
                            entry = {"file_name": new_file_name,"height": height, "width": width, "id": unique_img_id, "patient_id": folder_name}
                            json_image_training.append(entry)
                            # json_image_training += '{"file_name":"' + new_file_name + '", "height": ' + str(
                            #     height) + ', "width": ' + str(width) + ', "id": ' + str(unique_img_id) + ', "patient_id": '+ str(folder_name) + '},'

                            area = 1
                            if len(bbox_coordinates)>0:
                                area = float(bbox_coordinates[2]) * float(bbox_coordinates[3])

                            entry_ann = {"image_id": unique_img_id, \
                                         "bbox": bbox_coordinates, \
                                         "patient_id": folder_name, \
                                         "id": unique_id , \
                                         "iscrowd": 0, \
                                         "area": area, \
                                         "category_id": cate}
                            json_annotations_training.append(entry_ann)
                            # json_annotations_training += '{"image_id": ' + str(unique_img_id) + ', \
                            #     "bbox": ' + str(bbox_coordinates) + \
                            #                     ', "patient_id": ' + str(folder_name) + \
                            #                     ', "id": ' + str(unique_id) + \
                            #                     ', "iscrowd": ' + str(0) + \
                            #                     ', "area": ' + str(area) + \
                            #                     ', "category_id": ' + str(cate) + '},'
                        elif dataset_type_ == 'val':
                            entry = {"file_name": new_file_name,"height": height, "width": width, "id": unique_img_id, "patient_id": folder_name}
                            json_image_val.append(entry)

                            area = 1
                            if len(bbox_coordinates)>0:
                                area = float(bbox_coordinates[2]) * float(bbox_coordinates[3])
                            
                            entry_ann = {"image_id": unique_img_id, \
                                         "bbox": bbox_coordinates, \
                                         "patient_id": folder_name, \
                                         "id": unique_id , \
                                         "iscrowd": 0, \
                                         "area": area, \
                                         "category_id": cate}
                            json_annotations_val.append(entry_ann)

                        elif dataset_type_ == 'test':
                            entry = {"file_name": new_file_name,"height": height, "width": width, "id": unique_img_id, "patient_id": folder_name}
                            json_image_test.append(entry)

                            area = 1
                            if len(bbox_coordinates)>0:
                                area = float(bbox_coordinates[2]) * float(bbox_coordinates[3])

                            entry_ann = {"image_id": unique_img_id, \
                                         "bbox": bbox_coordinates, \
                                         "patient_id": folder_name, \
                                         "id": unique_id , \
                                         "iscrowd": 0, \
                                         "area": area, \
                                         "category_id": cate}
                            json_annotations_test.append(entry_ann)

        f.close()
        f2.close()
        f3.close()
        f4.close()

        # json_image_training = json_image_training[:-1]
        # json_image_training += '],'
        # json_annotations_training = json_annotations_training[:-1]
        # json_annotations_training += '] }'
        json_image_training = {"images": json_image_training}
        json_annotations_training = {"annotations": json_annotations_training}

        json_image_val = {"images": json_image_val}
        json_annotations_val = {"annotations": json_annotations_val}

        json_image_test = {"images": json_image_test}
        json_annotations_test = {"annotations": json_annotations_test}

        # json_image_test = json_image_test[:-1]
        # json_image_test += '],'
        # json_annotations_test = json_annotations_test[:-1]
        # json_annotations_test += '] }'

        # json_image_val = json_image_val[:-1]
        # json_image_val += '],'
        # json_annotations_val = json_annotations_val[:-1]
        # json_annotations_val += '] }'
        # print(json_image)
        # print(json_annotations)
        # print(i)
        json_everything_training = {**json_categories_training, **json_image_training, **json_annotations_training}
        json_everything_val = {**json_categories_val, **json_image_val, **json_annotations_val}
        json_everything_test = {**json_categories_test, **json_image_test, **json_annotations_test}
        
        # print(json_everything_training)
        np.save(os.path.join(self.output_path, 'json_everything_training.npy'), json_everything_training)
        np.save(os.path.join(self.output_path, 'json_everything_test.npy'), json_everything_test)
        np.save(os.path.join(self.output_path, 'json_everything_val.npy'), json_everything_val)

        # json_object_training = json.loads(json.dumps(json_everything_training))
        # json_object_test = json.loads(json_everything_test)
        # json_object_val = json.loads(json_everything_val)
        # Writing to sample.json
        with open("train_cvam.json", "w") as outfile1:
            outfile1.write(json.dumps(json_everything_training))

        with open("val_cvam.json", "w") as outfile2:
            outfile2.write(json.dumps(json_everything_val))

        with open("tests_cvam.json", "w") as outfile3:
            outfile3.write(json.dumps(json_everything_test))
        


def main():
    parser = argparse.ArgumentParser(description="Image pre-processing for VinDr Mammo dataset")
    parser.add_argument("-l", "--csv_path", type=str, default="./finding_annotations.csv", help="Path to csv file")
    parser.add_argument("-d", "--dataset_path", type=str, default="/home/rwl/Downloads/Dataset/physionet.org/files/vindr-mammo/1.0.0/images/", help="Path to dataset")
    parser.add_argument("-t", "--missing_img_txt", type=str, default="./missing_images_cvam.txt",
                        help="Path to text file to store path of images that were not found")
    parser.add_argument("-i", "--processing_issue_img_txt", type=str, default="./processing_issue_img_txt_cvam.txt",
                        help="Path to text file to store path of images that have some issue during processing")
    parser.add_argument("-p", "--processed_img_txt", type=str, default="./processed_images_cvam.txt",
                        help="Path to text file to store path of processed images")
    parser.add_argument("-o", "--output_path", type=str, default="/home/rwl/Downloads/Dataset/physionet.org/files/vindr-mammo/1.0.0/processed_imgs_cvam/",
                        help="Path to output directory")
    parser.add_argument("-dt", "--dataset_type", type=str, default="training", choices=['training', 'test'],
                        help="testing or training")

    args = parser.parse_args()

    pre_processing = PreProcessing(args)
    pre_processing.run()


if __name__ == "__main__":
    main()