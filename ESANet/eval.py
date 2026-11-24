# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

IZMJENE za BRANCH skup:

1. Dodan --split (zadano 'test'). Bez njega prepare_data pada na valid_set
   koji je za branch 'valid', pa bi se mjerilo na 80 validacijskih slika
   umjesto na 77 testnih.

2. Ispisuje IoU po klasi i mIoU BEZ pozadine.
   Njihova konvencija prosjecuje preko svih klasa osim voida, a kod BRANCH
   skupa je pozadina stvarna klasa (76 % piksela, IoU ~0.99), pa njihov
   mIoU izlazi nekoliko bodova visi. Za usporedbu s ostatkom projekta
   koristi se brojka bez pozadine.
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src.metric import MeanIntersectionOverUnion
from src.prepare_data import prepare_data


if __name__ == '__main__':
    # arguments
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Evaluation)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--ckpt_path', type=str,
                        required=True,
                        help='Path to the checkpoint of the trained model.')
    parser.add_argument('--split', type=str, default='test',
                        choices=['valid', 'test'],
                        help='Na kojem splitu evaluirati.')
    args = parser.parse_args()

    # dataset
    args.pretrained_on_imagenet = False  # we are loading other weights anyway
    _, data_loader, *add_data_loader = prepare_data(args,
                                                    with_input_orig=True,
                                                    split=args.split)
    if args.valid_full_res:
        # cityscapes only -> use dataloader that returns full resolution images
        data_loader = add_data_loader[0]

    n_classes = data_loader.dataset.n_classes_without_void

    # model and checkpoint loading
    model, device = build_model(args, n_classes=n_classes)
    checkpoint = torch.load(args.ckpt_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint from {}'.format(args.ckpt_path))

    model.eval()
    model.to(device)

    n_samples = 0
    metrics = {}

    torch.set_grad_enabled(False)

    for camera in data_loader.dataset.cameras:
        metrics[camera] = MeanIntersectionOverUnion(n_classes=n_classes)
        with data_loader.dataset.filter_camera(camera):
            for sample in tqdm(data_loader, desc=camera, unit='batch'):
                n_samples += sample['image'].shape[0]

                image = sample['image'].to(device)
                depth = sample['depth'].to(device)
                label_orig = sample['label_orig']

                _, image_h, image_w = label_orig.shape

                if args.modality == 'rgbd':
                    inputs = (image, depth)
                elif args.modality == 'rgb':
                    inputs = (image,)
                elif args.modality == 'depth':
                    inputs = (depth,)

                pred = model(*inputs)

                pred = F.interpolate(pred, (image_h, image_w),
                                     mode='bilinear',
                                     align_corners=False)
                pred = torch.argmax(pred, dim=1)

                # ignore void pixels
                mask = label_orig > 0
                label = torch.masked_select(label_orig, mask)
                pred = torch.masked_select(pred, mask.to(device))

                # In the label 0 is void but in the prediction 0 is wall.
                # In order for the label and prediction indices to match
                # we need to subtract 1 of the label.
                label -= 1

                pred = pred.to('cpu')
                metrics[camera].update(pred, label)

        miou = metrics[camera].compute()
        print(f'Camera: {camera} mIoU: {100*miou:0.2f}')

    # sum confusion matrices of all cameras
    metrics['all'] = MeanIntersectionOverUnion(n_classes)
    for camera in data_loader.dataset.cameras:
        metrics['all'].update_cm(metrics[camera].cm)

    miou, ious = metrics['all'].compute(return_ious=True)
    imena = list(data_loader.dataset.class_names_without_void)
    ious_np = np.array([float(x) for x in ious], dtype=np.float64)

    print()
    print(f'Split: {args.split}   slika: {n_samples}   klasa: {n_classes}')
    print()
    print('IoU po klasi:')
    for i, ime in enumerate(imena):
        print(f'  {ime:14} {100*ious_np[i]:6.2f}')

    # Pozadina je kod BRANCH skupa indeks 0 nakon label -= 1.
    # Ako prvi razred nije 'background', nema sto izbacivati.
    if imena and imena[0] == 'background':
        bez_poz = float(np.nanmean(ious_np[1:]))
    else:
        bez_poz = float('nan')
        print("\n  (prvi razred nije 'background' — mIoU bez pozadine se ne racuna)")

    print()
    print(f'mIoU (sve klase, njihova konvencija): {100*float(miou):0.2f}')
    if not np.isnan(bez_poz):
        print(f'mIoU BEZ pozadine (za usporedbu):     {100*bez_poz:0.2f}')
    print()
    print('Za rad se koristi brojka bez pozadine — tako racunaju i ostali '
          'modeli u projektu.')