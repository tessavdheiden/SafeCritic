import argparse
import shutil

parser = argparse.ArgumentParser()

# Example of dir_dataset: /home/userName/Desktop/FLORA/code/social_gan/datasets/dataset/UCY/
parser.add_argument('--dir_dataset', type=str, help="directory where all scene folders are")
parser.add_argument('--scene', type=str, help="name of the scene in which training files will be deleted")


def delete_training_files(dir_dataset, scene):
    training_path = dir_dataset + scene + "/Training/"
    shutil.rmtree(training_path + "train", ignore_errors=True)
    shutil.rmtree(training_path + "val", ignore_errors=True)
    shutil.rmtree(training_path + "test", ignore_errors=True)


def main(args):
    delete_training_files(args.dir_dataset, args.scene)
    return True


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)