import os
import argparse


parser = argparse.ArgumentParser(description='File Name modification')
parser.add_argument('--file_path', required=True, help='Source image path')
args = parser.parse_args()

print("File Name Modification process starting..")

file_path = args.file_path
file_names = os.listdir(file_path)

i = 1
for name in file_names:
    src = os.path.join(file_path, name)
    dst = str(i) + '.jpg'
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1
    print(src," to ",dst)

print("Done!")