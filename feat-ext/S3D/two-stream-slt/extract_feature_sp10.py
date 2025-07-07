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
import json


language_codes_sign = {
    'zh': 'CSL',
    'uk': 'UKL',
    'ru': 'RSL',
    'bg': 'BQN',
    'is': 'ICL',
    'de': 'GSG',
    'it': 'ISE',
    'sv': 'SWL',
    'lt': 'LLS',
    'en': 'BFI'
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


def load_rgb_frames_from_video(vid_path, logger, resize=(224, 224), desired_channel_order='rgb'):
    vidcap = cv2.VideoCapture(vid_path)
    
    # Check if the video opened successfully
    if not vidcap.isOpened():
        logger.error(f'Error opening video file {vid_path}')
        return

    frames = []

    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for offset in range(total_frames):
        success, img = vidcap.read()

        w, h, c = img.shape
        w_new, h_new = resize
        if w < 226 or h < 226:
            d = 226. - min(w, h)
            sc = 1 + d / min(w, h)
            img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        if w > w_new or h > h_new:
            img = cv2.resize(img, (math.ceil(w * (w_new / w)), math.ceil(h * (h_new / h))))

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
    
    for root, annot in zip(frame_roots, annotations):
        logger.info(f"Extracting features for {root}")
        with open(annot) as f:
            data = json.load(f)
        
        out_results = []
        for subdir in glob.glob(os.path.join(root, "*")):
            for video_path in glob.glob(os.path.join(subdir, "*")):
                video_id = int(video_path.split('/')[-2])  # Extract the ID from the folder structure
                dev_entry = next((entry for entry in data if entry['id'] == video_id), None)

                if dev_entry:
                    sign_list = [l for l in dev_entry['sign_list'] if l['lang']==video_path.split('/')[-1][:2]]
                    lang = sign_list[0]['lang']  
                    sign_lang = language_codes_sign[lang]
                    text = sign_list[0]['text']
                    frames = load_rgb_frames_from_video(video_path, logger, desired_channel_order=inp_channels)
                    features = extract_s3d_features(s3d, frames, device, logger)
                    output = {
                        'name': video_path[3:],
                        'signer': '',
                        'gloss': '',
                        'text': ' '.join(list(text)) if lang == 'zh' else text,
                        'sign': features.detach().cpu().numpy(),
                        'lang': lang,
                        'sign_lang': sign_lang
                    }
                    # print(output)
                    logger.info(f"Extracted features for {output['name']} with shape {output['sign'].shape}")
                    out_results.append(output)
            #     break
            # break
        save_features_to_file(out_results, os.path.join(outroot, os.path.basename(root) + '_s3d.npz'))
            

if __name__ == '__main__':
    print("Cuda available: ", torch.cuda.is_available())

    # ======= Extract Features for SP-10  ========
    frame_roots = [
        'SP-10-dataset/videos/train',
        'SP-10-dataset/videos/dev',
        'SP-10-dataset/videos/test'
    ]

    annotations = [
        'SP-10-dataset/dataset/train.json',
        'SP-10-dataset/dataset/dev.json',
        'SP-10-dataset/dataset/test.json'
    ]

    out = 'feature_ext/data/s3d-features/sp-10'

    run(frame_roots, annotations, out, 'rgb')
