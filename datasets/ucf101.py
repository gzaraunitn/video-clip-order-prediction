"""Dataset utils for NN."""
import os
import random
from glob import glob
from pprint import pprint
import uuid
import tempfile

import numpy as np
import ffmpeg
import skvideo.io
import pandas as pd
from skvideo.io import ffprobe
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

from os import listdir
from os.path import join
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", str(text))]


class UCF101VCOPDataset(Dataset):
    """UCF101 dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    Need the corresponding configuration file exists.

    Args:
        root_dir (string): Directory with videos and splits.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 16/32/64.
        interval (int): number of frames between clips, 16/32.
        tuple_len (int): number of clips in each tuple, 3/4/5.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """

    def __init__(
        self,
        file_list,
        root_dir,
        clip_len,
        interval,
        tuple_len,
        train=True,
        transforms_=None,
        epic_kitchens=False,
        remove_target_private=True,
        n_classes=12
    ):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.interval = interval
        self.tuple_len = tuple_len
        self.train = train
        self.transforms_ = transforms_
        self.toPIL = transforms.ToPILImage()
        self.tuple_total_frames = clip_len * tuple_len + interval * (tuple_len - 1)

        self.epic_kitchens = epic_kitchens

        # if self.train:
        #     vcop_train_split_name = "vcop_train_{}_{}_{}.txt".format(
        #         clip_len, interval, tuple_len
        #     )
        #     vcop_train_split_path = os.path.join(
        #         root_dir, "split", vcop_train_split_name
        #     )
        #     self.train_split = pd.read_csv(vcop_train_split_path, header=None)[0]
        # else:
        #     vcop_test_split_name = "vcop_test_{}_{}_{}.txt".format(
        #         clip_len, interval, tuple_len
        #     )
        #     vcop_test_split_path = os.path.join(root_dir, "split", vcop_test_split_name)
        #     self.test_split = pd.read_csv(vcop_test_split_path, header=None)[0]

        self.videos_with_class = []

        with open(file_list, "r") as filelist:
            for line in filelist:
                split_line = line.split()
                path = split_line[0]

                if self.epic_kitchens:
                    start_frame = int(split_line[1])
                    stop_frame = int(split_line[2])
                    label = int(split_line[3])
                    if remove_target_private:
                        if label < n_classes:
                            if len(self.find_frames(path)) > self.tuple_total_frames:
                                self.videos_with_class.append(
                                    (path, start_frame, stop_frame, label)
                                )
                    else:
                        if len(self.find_frames(path)) > self.tuple_total_frames:
                            self.videos_with_class.append(
                                (path, start_frame, stop_frame, label)
                            )
                else:
                    label = int(split_line[1])
                    if remove_target_private:
                        if label < n_classes:
                            if len(self.find_frames(path)) > self.tuple_total_frames:
                                self.videos_with_class.append((path, label))
                    else:
                        if len(self.find_frames(path)) > self.tuple_total_frames:
                            self.videos_with_class.append((path, label))

    def __len__(self):
        if self.train:
            return len(self.videos_with_class)
        else:
            return len(self.videos_with_class)

    def is_img(self, f):
        return str(f).lower().endswith("jpg") or str(f).lower().endswith("jpeg")

    def find_frames(self, video):
        """Finds frames from input sequence."""
        frames = [join(video, f) for f in listdir(video) if self.is_img(f)]
        return frames

    def load_frame(self, path, mode="RGB"):
        frame = Image.open(path).convert(mode)
        frame = TF.to_tensor(frame)
        return frame

    def __getitem__(self, idx):
        """
        Returns:
            tuple_clip (tensor): [tuple_len x channel x time x height x width]
            tuple_order (tensor): [tuple_len]
        """
        if self.epic_kitchens:
            video, start_frame, stop_frame, y = self.videos_with_class[idx]
        else:
            video, y = self.videos_with_class[idx]

        #filename = os.path.join(self.root_dir, "video", videoname)
        # videodata = skvideo.io.vread(filename)
        videodata = self.find_frames(video)
        videodata.sort(key=natural_keys)
        length = len(videodata)

        tuple_clip = []
        tuple_order = list(range(0, self.tuple_len))

        # random select tuple for train, deterministic random select for test
        if self.train:
            tuple_start = random.randint(0, length - self.tuple_total_frames)
        else:
            random.seed(idx)
            tuple_start = random.randint(0, length - self.tuple_total_frames)

        clip_start = tuple_start
        for _ in range(self.tuple_len):
            clip_str = videodata[clip_start : clip_start + self.clip_len]
            clip = [self.load_frame(f) for f in clip_str]
            tuple_clip.append(clip)
            clip_start = clip_start + self.clip_len + self.interval

        clip_and_order = list(zip(tuple_clip, tuple_order))
        # random shuffle for train, the same shuffle for test
        if self.train:
            random.shuffle(clip_and_order)
        else:
            random.seed(idx)
            random.shuffle(clip_and_order)
        tuple_clip, tuple_order = zip(*clip_and_order)

        if self.transforms_:
            trans_tuple = []
            for clip in tuple_clip:
                trans_clip = []
                # fix seed, apply the sample `random transformation` for all frames in the clip
                seed = random.random()
                for frame in clip:
                    random.seed(seed)
                    frame = self.toPIL(frame)  # PIL image
                    frame = self.transforms_(frame)  # tensor [C x H x W]
                    trans_clip.append(frame)
                # (T x C X H x W) to (C X T x H x W)
                trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                trans_tuple.append(trans_clip)
            tuple_clip = trans_tuple
        else:
            tuple_clip = [torch.tensor(clip) for clip in tuple_clip]

        return torch.stack(tuple_clip), torch.tensor(tuple_order)


def export_tuple(tuple_clip, tuple_order, dir):
    """export tuple_clip and set its name with correct order.

    Args:
        tuple_clip (tensor): [tuple_len x channel x time x height x width]
        tuple_order (tensor): [tuple_len]
    """
    tuple_len, channel, time, height, width = tuple_clip.shape
    for i in range(tuple_len):
        filename = os.path.join(dir, "c{}.mp4".format(tuple_order[i]))
        skvideo.io.vwrite(filename, tuple_clip[i])


def gen_ucf101_vcop_splits(root_dir, clip_len, interval, tuple_len):
    """Generate split files for different configs."""
    vcop_train_split_name = "vcop_train_{}_{}_{}.txt".format(
        clip_len, interval, tuple_len
    )
    vcop_train_split_path = os.path.join(root_dir, "split", vcop_train_split_name)
    vcop_test_split_name = "vcop_test_{}_{}_{}.txt".format(
        clip_len, interval, tuple_len
    )
    vcop_test_split_path = os.path.join(root_dir, "split", vcop_test_split_name)
    # minimum length of video to extract one tuple
    min_video_len = clip_len * tuple_len + interval * (tuple_len - 1)

    def _video_longer_enough(filename):
        """Return true if video `filename` is longer than `min_video_len`"""
        path = os.path.join(root_dir, "video", filename)
        metadata = ffprobe(path)["video"]
        return eval(metadata["@nb_frames"]) >= min_video_len

    train_split = pd.read_csv(
        os.path.join(root_dir, "split", "trainlist01.txt"), header=None, sep=" "
    )[0]
    train_split = train_split[train_split.apply(_video_longer_enough)]
    train_split.to_csv(vcop_train_split_path, index=None)

    test_split = pd.read_csv(
        os.path.join(root_dir, "split", "testlist01.txt"), header=None, sep=" "
    )[0]
    test_split = test_split[test_split.apply(_video_longer_enough)]
    test_split.to_csv(vcop_test_split_path, index=None)


def ucf101_stats():
    """UCF101 statistics"""
    collects = {
        "nb_frames": [],
        "heights": [],
        "widths": [],
        "aspect_ratios": [],
        "frame_rates": [],
    }

    for filename in glob("../data/ucf101/video/*/*.avi"):
        metadata = ffprobe(filename)["video"]
        collects["nb_frames"].append(eval(metadata["@nb_frames"]))
        collects["heights"].append(eval(metadata["@height"]))
        collects["widths"].append(eval(metadata["@width"]))
        collects["aspect_ratios"].append(metadata["@display_aspect_ratio"])
        collects["frame_rates"].append(eval(metadata["@avg_frame_rate"]))

    stats = {key: sorted(list(set(collects[key]))) for key in collects.keys()}
    stats["nb_frames"] = [stats["nb_frames"][0], stats["nb_frames"][-1]]

    pprint(stats)


if __name__ == "__main__":
    seed = 632
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ucf101_stats()
    # gen_ucf101_vcop_splits('../data/ucf101', 16, 16, 2)
    # gen_ucf101_vcop_splits('../data/ucf101', 16, 32, 3)
    gen_ucf101_vcop_splits("../data/ucf101", 16, 8, 3)

    # train_transforms = transforms.Compose([
    #     transforms.Resize((128, 171)),
    #     transforms.RandomCrop(112),
    #     transforms.ToTensor()])
    # # train_dataset = UCF101FOPDataset('../data/ucf101', 8, 3, True, train_transforms)
    # # train_dataset = UCF101VCOPDataset('../data/ucf101', 16, 8, 3, True, train_transforms)
    # train_dataset = UCF101Dataset('../data/ucf101', 16, False, train_transforms)
    # # train_dataset = UCF101RetrievalDataset('../data/ucf101', 16, 10, True, train_transforms)
    # train_dataloader = DataLoader(train_dataset, batch_size=8)

    # for i, data in enumerate(train_dataloader):
    #     clips, idxs = data
    #     # for i in range(10):
    #     #     filename = os.path.join('{}.mp4'.format(i))
    #     #     skvideo.io.vwrite(filename, clips[0][i])
    #     print(clips.shape)
    #     print(idxs)
    #     exit()
    # pass
