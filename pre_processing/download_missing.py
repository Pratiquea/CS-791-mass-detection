import argparse
import os
import subprocess

class DownloadMissing(object):
    def __init__(self, args):
        self.txt_file = args.input_txt_file
        self.output_dir = args.output_dir
        self.download()

    def download(self):
        cmd_part_1 = 'wget -r -N -c -np --user pratique --ask-password '
        prefix = 'https://physionet.org/files/vindr-mammo/1.0.0/images/'
        # /d94fa669c37cc757ea519ca3c291ccb4/84327b97a8195be40744b4983d5f7cbe.dicom
        dataset_dir = "physionet.org/files/vindr-mammo/1.0.0/images/"
        all_files = ''
        with open(self.txt_file, 'r') as f:
            count = 0
            for line in f:
                # print(line)
                line = line.strip().split('/')
                file_to_download = line[-1]
                file_dir = line[-2]
                print(line)
                if count == 2:
                    break
                full_dir_path= os.path.join(self.output_dir, dataset_dir, file_dir)
                if not os.path.exists(full_dir_path):
                    os.makedirs(full_dir_path)
                download_path = os.path.join(prefix, file_dir ,file_to_download)
                all_files += download_path + ' '
                # if not os.path.exists(os.path.join(self.output_dir, line)):
                #     print(line)
                #     os.system('wget -P {} {}'.format(self.output_dir, line))
                count += 1
        print(all_files)
        entire_cmd = cmd_part_1 + all_files
        # import sys
        # sys.exit()
        os.chdir(self.output_dir)
        cwd = os.getcwd()
        os.system(entire_cmd)

def main():
    parser = argparse.ArgumentParser(description="Download missing mammograms images")
    parser.add_argument("-t","--input_txt_file", type=str, default = "./missing_images.txt", help="Path to the text file containing the list of images to download")
    parser.add_argument("-o","--output_dir", type=str, default="/home/rwl/Downloads/Dataset/", help="Path to the directory where the images will be downloaded")
    args = parser.parse_args()
    downloder=DownloadMissing(args)


if __name__ == "__main__":
    main()
