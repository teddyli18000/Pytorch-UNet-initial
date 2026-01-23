import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet.unet_model import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')  # no longer required, interactive fallback supported
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def _interactive_input_files(prompt_text="请输入要预测的图像文件路径（多个用空格分隔）："):
    s = input(prompt_text).strip()
    if not s:
        return []
    parts = s.split()
    return parts


def _interactive_model_file(prompt_text="请输入模型文件（.pth）的路径："):
    s = input(prompt_text).strip()
    return s


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Interactive fallback for input images
    if not args.input:
        logging.info("没有从命令行收到输入图像路径，将使用交互式输入。")
        inputs = _interactive_input_files()
        if not inputs:
            logging.error("未提供任何输入图像路径，程序退出。")
            sys.exit(1)
        args.input = inputs

    # Interactive fallback for model path (if default MODEL.pth is left or file doesn't exist)
    if args.model == 'MODEL.pth' or not os.path.exists(args.model):
        logging.info("模型文件不存在或使用默认名，进入交互式模型路径输入。")
        model_path = _interactive_model_file()
        if not model_path:
            logging.error("未提供模型文件路径，程序退出。")
            sys.exit(1)
        args.model = model_path

    # Validate input files exist
    missing_inputs = [p for p in args.input if not os.path.exists(p)]
    if missing_inputs:
        logging.error("以下输入文件不存在：")
        for p in missing_inputs:
            logging.error(f"  {p}")
        logging.error("请检查路径（支持相对路径和绝对路径），然后重试。")
        sys.exit(1)

    # If outputs provided, make sure length matches inputs
    if args.output and len(args.output) != len(args.input):
        logging.error("提供了 --output，但输出文件数量与输入文件数量不匹配。")
        sys.exit(1)

    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    try:
        net.to(device=device)
        state_dict = torch.load(args.model, map_location=device)
    except Exception as e:
        logging.error(f'加载模型时发生错误: {e}')
        sys.exit(1)

    mask_values = state_dict.pop('mask_values', [0, 1]) if isinstance(state_dict, dict) else [0, 1]
    try:
        net.load_state_dict(state_dict)
    except Exception as e:
        logging.error(f'将权重加载到网络时出错: {e}')
        sys.exit(1)

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        try:
            img = Image.open(filename)
        except Exception as e:
            logging.error(f'打开图像 {filename} 失败: {e}')
            continue

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = out_files[i]
            try:
                result = mask_to_image(mask, mask_values)
                # Ensure output directory exists
                out_dir = os.path.dirname(out_filename)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                result.save(out_filename)
                out_abs = os.path.abspath(out_filename)
                logging.info(f'Mask saved to {out_abs}')
                # also print for immediate visibility
                print(f'已保存: {out_abs}')
            except Exception as e:
                logging.error(f'保存结果到 {out_filename} 失败: {e}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
