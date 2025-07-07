import warnings, pickle, gzip, argparse, os, sys
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings("ignore")
from modelling.model import build_model
from modelling.S3D import S3D_backbone

sys.path.append(os.getcwd())#slt dir

from utils.misc import (
    get_logger, load_config, make_logger, 
    init_DDP, move_to_device,
    synchronize 
)
from dataset.Dataloader import build_dataloader
from utils.progressbar import ProgressBar




if __name__ == "__main__":
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
    
    #load model
    model = S3D_backbone(in_channel=3, **cfg['model']['RecognitionNetwork']['s3d']).to(cfg['device'])

    for split in args.output_split.split(','):
        logger.info('Extract visual feature on {} set'.format(split))
        random_tensor = torch.randn(1,3,32, 224,224).to(cfg['device'])
        output = model(random_tensor, sgn_lengths=torch.tensor([32]).to(cfg['device']))
        print(output['sgn'].shape)
                   
