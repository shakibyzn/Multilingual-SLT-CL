import warnings, argparse, sys

warnings.filterwarnings("ignore")
from modelling.S3D import S3D_backbone

# sys.path.append(os.getcwd())#slt dir

from utils.misc import load_config, make_logger
import os

import numpy as np
import torch
import cv2
import argparse
from tqdm import tqdm
import glob
import pandas as pd
import gzip
import pickle
import math


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
    image_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
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

def extract_s3d_features(model, frames, device, logger):
    # T, H, W, C to B, C, T, H, W
    # logger.info(f"Extracting features for {frames.shape}")
    frames = torch.from_numpy(frames)
    frames = frames.permute(3, 0, 1, 2).unsqueeze(0)
    frames = frames.to(device)
    sgn_lengths = torch.tensor([frames.shape[2]]).to(device)
    with torch.no_grad():
        outputs = model(frames, sgn_lengths)
        
    # logger.info(f"Extracted features with shape {outputs['sgn'].shape}")
    return outputs['sgn'].squeeze(0)
    

def run(frame_roots, annotations, outroot, inp_channels='rgb'):
    root = 'feature_ext'
    parser = argparse.ArgumentParser("SLT baseline Testing")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        '--output_split',
        default='dev,test,train',
        type=str,
        help='sep by ,'
    )
    parser.add_argument(
        '--output_subdir',
        default='extract_feature',
        type=str
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    args.outputdir=os.path.join(cfg['training']['model_dir'],args.output_subdir)
    logger = make_logger(model_dir=args.outputdir, log_file=f"prediction.log")
    
    # ===== setup models ======
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    #load model
    s3d = S3D_backbone(in_channel=3, **cfg['model']['RecognitionNetwork']['s3d'])

    s3d.to(device)
    # i3d = nn.DataParallel(i3d)

    print('feature extraction starts.')
    s3d.train(False)  # Set model to evaluate mode
    
    for set_type in args.output_split.split(','):
        logger.info(f"Extracting features for {set_type} set")
        out_results = []
        for item in tqdm(frame_roots['info']):
            # print(annotations[annotations['name'] == item['name']]['split'].values)
            item_split = annotations[annotations['name'] == item['name']]['split']
            item_split = item_split.tolist().pop() if len(item_split) > 0 else None
            # check if not empty
            if item_split != None and item_split == set_type:
                name = item['name']
                text = ' '.join(item['label_char'])
                gloss = ' '.join(item['label_gloss'])
                signer = str(item['signer'])
                video_name = os.path.join(root, 'CSL-Daily/sentence/frames_512x512', name)

                frames = load_all_rgb_frames_from_folder(video_name, inp_channels)
                features = extract_s3d_features(s3d, frames, device, logger)
                # features_torch = torch.stack(features)
                output = {
                    "name": os.path.basename(video_name).strip(),
                    "sign": features.detach().cpu().numpy(),
                    "signer": signer,
                    "gloss": gloss,
                    "text": text,
                    'lang': 'zh',
                    'sign_lang': LANG_MAPPER['zh']
                }
                # print(output)
                logger.info(f"Extracted features for {output['name']} with shape {output['sign'].shape}")
                out_results.append(output)
            
        save_features_to_file(out_results, os.path.join(outroot, set_type + '_s3d.npz'))
            

if __name__ == '__main__':
    print("Cuda available: ", torch.cuda.is_available())

    # ======= Extract Features for CSL-Daily ========
    root = 'feature_ext'
    with open(os.path.join(root, 'CSL-Daily/sentence_label/csl2020ct_v2.pkl'), 'rb') as f: 
        frame_roots = pickle.load(f) 
    
    annotations = pd.read_csv(os.path.join(root, 'CSL-Daily/sentence_label/split_1.txt'), sep='|')

    out = 'feature_ext/data/s3d-features/csl-daily'

    run(frame_roots, annotations, out, 'rgb')
