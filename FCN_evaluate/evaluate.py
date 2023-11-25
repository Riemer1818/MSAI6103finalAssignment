import os
import torch
import torchvision.transforms as T
import argparse
import numpy as np
from PIL import Image
from util import segrun, fast_hist, get_scores
from cityscapes import cityscapes

parser = argparse.ArgumentParser()
parser.add_argument("--cityscapes_dir", type=str, required=True, help="Path to the original cityscapes dataset")
parser.add_argument("--result_dir", type=str, required=True, help="Path to the generated images to be evaluated")
parser.add_argument("--output_dir", type=str, required=True, help="Where to save the evaluation results")
parser.add_argument("--pytorch_model_path", type=str, required=True, help="Path to the PyTorch model")
parser.add_argument("--device", type=str, default='cuda:0', help="Which device to use ('cuda:0' for GPU or 'cpu' for CPU)")
parser.add_argument("--split", type=str, default='val', help="Data split to be evaluated")
parser.add_argument("--save_output_images", type=int, default=0, help="Whether to save the output images")
args = parser.parse_args()

def main():
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if args.save_output_images > 0:
        output_image_dir = args.output_dir + 'image_outputs/'
        if not os.path.isdir(output_image_dir):
            os.makedirs(output_image_dir)
    CS = cityscapes(args.cityscapes_dir)
    n_cl = len(CS.classes)
    label_frames = CS.list_label_frames(args.split)
    
    device = torch.device(args.device)
    model = torch.load(args.pytorch_model_path)
    model.to(device).eval()

    hist_perframe = np.zeros((n_cl, n_cl))
    for i, idx in enumerate(label_frames):
        if i % 10 == 0:
            print('Evaluating: %d/%d' % (i, len(label_frames)))
        city = idx.split('_')[0]
        # idx is city_shot_frame
        label = CS.load_label(args.split, city, idx)
        im_file = args.result_dir + '/' + idx + '_leftImg8bit.png'
        im = np.array(Image.open(im_file))
        im = np.array(Image.fromarray(im).resize((label.shape[1], label.shape[2])))  # Replace deprecated scipy.misc.imresize
        out = segrun(model, device, CS.preprocess(im))  # Assuming segrun now accepts PyTorch model and device

        hist_perframe += fast_hist(label.flatten(), out.flatten(), n_cl)
        if args.save_output_images > 0:
            label_im = CS.palette(label)
            pred_im = CS.palette(out)
            Image.fromarray(pred_im).save(output_image_dir + '/' + str(i) + '_pred.jpg')
            Image.fromarray(label_im).save(output_image_dir + '/' + str(i) + '_gt.jpg')
            Image.fromarray(im).save(output_image_dir + '/' + str(i) + '_input.jpg')

    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
    with open(args.output_dir + '/evaluation_results.txt', 'w') as f:
        f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc)
        f.write('Mean class accuracy: %f\n' % mean_class_acc)
        f.write('Mean class IoU: %f\n' % mean_class_iou)
        f.write('************ Per class numbers below ************\n')
        for i, cl in enumerate(CS.classes):
            while len(cl) < 15:
                cl = cl + ' '
            f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))
            
main()