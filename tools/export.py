from detectron2.config import get_cfg
from stoma.data import DatasetMapper, builtin
from stoma.modeling import KRCNNConvHead
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import (DefaultTrainer, default_argument_parser,
                               default_setup, hooks, launch)

from tools.train_net import Trainer
import torch

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()

    # a dirty fix for the keypoint resolution config
    cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = (14, 14)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def load(args):
    cfg = setup(args)
    builtin.register_stoma(args.dataset_dir)
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=args.resume
    )
    return model

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--dataset-dir", default="datasets")
    parser.add_argument("--output")
    args = parser.parse_args()
    print("Command Line Args:", args)

    model = load(args)
    model.eval()

    INPUT_H = 2048
    INPUT_W = 2880
    BATCH_SIZE = 1
    CHANNELS = 3

    dummy_input = torch.randn(CHANNELS,
                            INPUT_H,
                            INPUT_W,
                            device='cpu')
    x = [{}]
    x[0]["image"] = dummy_input

    torch.onnx.export(
        model,
        x,
        args.output,
        verbose=True,
    )
    