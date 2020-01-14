"""
Script to download the Baseball Database (BBDB).
"""
import os
import json
import collections
import subprocess
import argparse


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--target-dir', dest='target_dir', default='./', 
        help="Target directory to download BBDB dataset")
    parser.add_argument('-i', '--input-json-file', dest='json_file', 
        default='./bbdb.v0.9.min.json', help='BBDB JSON file')
    parser.add_argument('-b', '--aria2-binary-path', dest='aria2_binary',
        default='aria2c', help='Specify path to aria2 binary')
    parser.add_argument('-n', '--number-of-files', dest='number_of_files', type=int,
        default=2000, help='Specify the number of files to download')
    args = parser.parse_args()

    return args

def create_dirs(dir_list):
    for directory in dir_list:
        if not os.path.isdir(directory):
            os.makedirs(directory)

def create_aria2_input_file(meta, fullgame_dir, aria2_input_file, number_of_files):
    with open(aria2_input_file, 'w') as fp:
        for gamecode in meta['database']:
            if number_of_files == 0:
                break
            videourl = meta['database'][gamecode]['videoUrl']
            filename = '{}.mp4'.format(gamecode)
            # Add to download list if the file does not exist
            if not os.path.isfile(os.path.join(fullgame_dir, filename)) or os.path.isfile("{}.aria2".format(os.path.join(fullgame_dir, filename))):
                print('{}\n  out={}'.format(videourl, filename), file=fp)
                number_of_files -= 1

def main():
    args = get_args()
    
    # Create directories
    bbdb_dir = os.path.join(args.target_dir, 'BBDB')
    fullgame_dir = os.path.join(bbdb_dir, 'fullgame')
    tmp_dir = os.path.join(bbdb_dir, 'tmp')
    dir_list = [fullgame_dir, tmp_dir]
    create_dirs(dir_list)

    # Read JSON
    with open(args.json_file, 'r') as fp:
        meta = json.load(fp, object_pairs_hook=collections.OrderedDict)

    # Create input file for aria2
    aria2_input_file = os.path.join(tmp_dir, 'bbdb_aria2_urls.txt')
    create_aria2_input_file(meta, fullgame_dir, aria2_input_file, args.number_of_files)

    command = [args.aria2_binary,
        '--max-connection-per-server=16',
        '--split=4',
        '--max-concurrent-downloads=4',
        '--continue',
        '--file-allocation=none',
        '--input-file={}'.format(aria2_input_file),
        '--dir={}'.format(fullgame_dir)]
    try:
        subprocess.call(command)
    except OSError as e:
        if e.errno == os.errno.ENOENT:
            print('\naria2c not found. Get aria2 from https://aria2.github.io\n'
                    'Tested version of aria2 is 1.33.0\n'
                    'You can specify path of aria2c by using -b option, '
                    'or --help for details.\n')
        else:
            raise


if __name__ == "__main__":
    main()
