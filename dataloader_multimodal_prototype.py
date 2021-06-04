# Author: Paritosh

import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
from opts import *
import pickle as pkl
import librosa


torch.manual_seed(random_seed); torch.cuda.manual_seed_all(random_seed); random.seed(random_seed); np.random.seed(random_seed)
torch.backends.cudnn.deterministic=True


def load_image(image_path, hori_flip, transform=None):
    image = Image.open(image_path)
    image = image.resize(c3d_input_resize, Image.BILINEAR)
    if hori_flip:
        image.transpose(Image.FLIP_LEFT_RIGHT)
    # image.show()
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image


def spec_to_image(spec, transform, eps=1e-6):
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    # print('diff: ', (spec_max - spec_min))
    if (spec_max - spec_min) == 0: # avoiding division by zero
        spec_scaled = 255 * (spec_norm - spec_min) / 5
    else:
        spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    if spec_scaled.shape[1] != audio_img_W:
        spec_scaled = spec_scaled[:,:audio_img_W]
    # spec_scaled = transform(spec_scaled).unsqueeze(0)
    true_image = Image.fromarray(spec_scaled)
    true_image = true_image.resize((224, 224), Image.BILINEAR)
    true_image = transform(true_image).unsqueeze(0)
    return true_image#spec_scaled


def get_melspectrogram_db(wav, sr, start_fr, end_fr, framerate, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    # print('audio file: ', file_path)
    # wav,sr = librosa.load(file_path, sr=sr)
    # print('sr = ', sr)
    # if sr != 44100:
    #     print('Different sr!')
    # print('framerate: ', framerate)
    start_time = int((start_fr/framerate)*sr)
    end_time = int((end_fr/framerate)*sr)
    # print('times: ', start_time, end_time)
    wav = wav[start_time:end_time]
    # print('wav shape: ', wav.shape)

    spec = librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft, hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    spec_db = librosa.power_to_db(spec,top_db=top_db)
    return spec_db


class VideoDataset(Dataset):
    def __init__(self, mode):
        super(VideoDataset, self).__init__()
        self.sampling_scheme = sampling_scheme
        self.mode = mode

        with open(anno_r2u_dir + '/annotations_' + self.sampling_scheme + '_' + self.mode + '.pkl', 'rb') as fp:
            self.set = pkl.load(fp)

        self.keys = list(self.set.keys())

        # print(len(self.set))


    def __getitem__(self, ix):
        # print('ix: ', ix)

        if backbone == 'R18-3D':
            transform = transforms.Compose([transforms.CenterCrop(c3d_H),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.43216, 0.39466, 0.37645], std=[0.22803, 0.22145, 0.21698])])
        elif backbone == 'C3D' or backbone == 'Dilated_C3D':
            transform = transforms.Compose([transforms.CenterCrop(c3d_H),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            input('Error: Unknown Backbone! What you want to do?')

        transform_audio_img = transforms.Compose([transforms.ToTensor()])


        sample = self.set[self.keys[ix]]
        # print(sample)
        sample_video = self.keys[ix][0]
        # print('sample_key: ', self.keys[ix])
        sample_frames = sample['frames']
        # print('frames: ', sample_frames)
        # print('reshaped frames: ', sample_frames.reshape((nclips, clip_len)))
        # print('first frame: ', sample_frames[0])
        sample_player_lvl = sample['player_level']
        sample_song_lvl = sample['song_level']
        sample_framerate = sample['framerate']
        image_list = sorted((glob.glob(os.path.join(dataset_video_dir + str(sample_video), '*.jpg'))))
        # print(image_list)

        # randomly applying hori_flip augmentation
        hori_flip = 0
        if self.mode == 'training':
            hori_flip = random.randint(0, 1)

        if with_modality_video:
            # following MTL-AQA style loading
            video = torch.zeros(nclips*clip_len, c3d_C, c3d_H, c3d_W)
            for frame in range(nclips*clip_len):
                video[frame] = load_image(image_list[sample_frames[frame]], hori_flip, transform)

        if with_modality_audio:
            sample_frames_reshaped = sample_frames.reshape((nclips, clip_len))

            audio_file = dataset_audio_dir + str(sample_video) + '.wav'
            wav, sr = librosa.load(audio_file, sr=None)
            # print('sr = ', sr)
            if sr != 44100:
                print('Different sr!')

            audio_imgs = torch.zeros(nclips, audio_img_C, audio_img_H, audio_img_W)

            for clip in range(nclips):
                start_fr = sample_frames_reshaped[clip][0]
                end_fr = sample_frames_reshaped[clip][-1]
                # print('start, end: ', start_fr, end_fr)
                mel_spec = get_melspectrogram_db(wav, sr=sr, start_fr=start_fr,
                                                 end_fr=end_fr, framerate=sample_framerate)
                # print('mel spec shape: ', mel_spec.shape)
                audio_imgs[clip] = spec_to_image(mel_spec, transform_audio_img)
                # print('Audio img shape: ', audio_img.shape)













        data = {}
        if with_modality_video:
            data['video'] = video
        if with_modality_audio:
            data['audio'] = audio_imgs
        data['player_lvl'] = sample_player_lvl
        return data


    def __len__(self):
        print(len(self.set))
        return len(self.set)
