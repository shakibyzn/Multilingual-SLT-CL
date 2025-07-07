import os

import numpy as np
import torch
import cv2
import logging
from pytorch_i3d import InceptionI3d
import argparse
from tqdm import tqdm
import glob
import pandas as pd
import gzip
import pickle
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

LANG_MAPPER = {
    'de': 'GSG',
    'zh': 'CSL',
    'en': 'ASE'
}

def save_features_to_file(features, filename):
    with gzip.open(filename, "wb") as f:
        pickle.dump(features, f)


def load_rgb_frames(image_dir, vid, start, num, desired_channel_order='rgb'):
    frames = []
    for i in range(start, start + num):
        img = cv2.imread(os.path.join(image_dir, vid, "images" + str(i).zfill(4) + '.png'))

        if desired_channel_order == 'bgr':
            img = img[:, :, [2, 1, 0]]

        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_all_rgb_frames_from_folder(folder, desired_channel_order='rgb'):
    frames = []
    image_files = [f for f in os.listdir(folder) if f.endswith('.png')]
    # print(sorted(image_files))
    # for fn in sorted(image_files, key=lambda x: int(x[6:6+4])):
    for fn in sorted(image_files):
    
        img = cv2.imread(os.path.join(folder, fn))
        img = cv2.resize(img, dsize=(224, 224))

        if desired_channel_order == 'bgr':
            img = img[:, :, [2, 1, 0]]

        img = (img / 255.) * 2 - 1
        frames.append(img)
    return np.asarray(frames, dtype=np.float32)


def load_rgb_frames_from_video(vid_path, vid, start, num, resize=(224, 224), desired_channel_order='rgb'):
    vidcap = cv2.VideoCapture(vid_path)

    frames = []

    total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for offset in range(total_frames):
        success, img = vidcap.read()

        w, h, c = img.shape
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > 256 or h > 256:
            img = cv2.resize(img, (math.ceil(w * (256 / w)), math.ceil(h * (256 / h))))

        img = (img / 255.) * 2 - 1

        if desired_channel_order == 'bgr':
            img = img[:, :, [2, 1, 0]]

        frames.append(img)

    return np.asarray(frames, dtype=np.float32)


def extract_features_fullvideo(model, inp, framespan, stride):
    rv = []

    indices = list(range(len(inp)))
    groups = []
    for ind in indices:

        if ind % stride == 0:
            groups.append(list(range(ind, ind+framespan)))

    for g in groups:
        # numpy array indexing will deal out-of-index case and return only till last available element
        frames = inp[g[0]: min(g[-1]+1, inp.shape[0])]

        num_pad = 9 - len(frames)
        if num_pad > 0:
            pad = np.tile(np.expand_dims(frames[-1], axis=0), (num_pad, 1, 1, 1))
            frames = np.concatenate([frames, pad], axis=0)

        frames = frames.transpose([3, 0, 1, 2])

        ft = _extract_features(model, frames)

        rv.append(ft)

    return rv


def _extract_features(model, frames):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inputs = torch.from_numpy(frames)

    inputs = inputs.unsqueeze(0)

    inputs = inputs.to(device)
    with torch.no_grad():
        ft = model.extract_features(inputs)
    # ft = ft.squeeze(-1).squeeze(-1)[0].transpose(0, 1)
    ft = ft.squeeze(-1).squeeze(-1).squeeze(-1)[0]

    ft = ft.cpu()

    return ft


def run(weight, frame_roots, annotations, outroot, inp_channels='rgb'):

    # ===== setup models ======
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    i3d = InceptionI3d(400, in_channels=3)

    i3d.replace_logits(2000)
    # i3d.replace_logits(1232)

    print('loading weights {}'.format(weight))
    # i3d.load_state_dict(torch.load(weight))
    i3d.load_state_dict(torch.load(weight, map_location=device))

    i3d.to(device)
    # i3d = nn.DataParallel(i3d)

    print('feature extraction starts.')
    i3d.train(False)  # Set model to evaluate mode
    
    framespan, stride = 8, 2
    
    for root, annot in zip(frame_roots, annotations):
        df = pd.read_csv(annot, sep="|")
        out_results = []
        for pth in tqdm(glob.glob(os.path.join(root, "*"))):

            frames = load_all_rgb_frames_from_folder(pth, inp_channels)
            features = extract_features_fullvideo(i3d, frames, framespan, stride)
            features_torch = torch.stack(features)
            output = {
                "name": os.path.basename(pth).strip(),
                "sign": features_torch.detach().cpu().numpy(),
                "signer": "",
                "gloss": df["orth"][df["name"] == os.path.basename(pth).strip()].values[0].lower(),
                "text": df["translation"][df["name"] == os.path.basename(pth).strip()].values[0].lower(),
                'lang': 'de',
                'sign_lang': LANG_MAPPER['de']
            }
            logging.info(f"Extracted features for {output['name']} with shape {output['sign'].shape}")
            out_results.append(output)
            
        save_features_to_file(out_results, os.path.join(outroot, os.path.basename(root) + '_i3d.npz'))
            

if __name__ == '__main__':
    print("Cuda available: ", torch.cuda.is_available())
    weight = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'

    # ======= Extract Features for PHEOENIX-2014-T ========
    frame_roots = [
        'RWTH-PHOENIX-Weather 2014 T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train',
        'RWTH-PHOENIX-Weather 2014 T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/dev',
        'RWTH-PHOENIX-Weather 2014 T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/test'
    ]
    
    annotations = [
        'RWTH-PHOENIX-Weather 2014 T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv',
        'RWTH-PHOENIX-Weather 2014 T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv',
        'RWTH-PHOENIX-Weather 2014 T/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv'
    ]

    out = 'data/i3d-features/phoenix'

    run(weight, frame_roots, annotations, out, 'rgb')
