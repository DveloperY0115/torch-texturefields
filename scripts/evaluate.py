"""
evaluate.py - Script used for evaluation.

This code is brought from the original implementation 
(https://github.com/autonomousvision/texture_fields) and slightly modified.

Unlike the original code, this script works only for objects in "cars" category.
"""

import argparse
import pandas as pd
import os
import sys

sys.path.append(".")
sys.path.append("..")

#from mesh2tex import config
from evaluation.eval import evaluate_generated_images

if __name__ == "__main__":
    if True:
        parser = argparse.ArgumentParser()
        parser.add_argument("--img_root", type=str, required=True, help="Directory storing images to be evaluated")
        parser.add_argument("--eval_out_dir", type=str, required=True, help="Directory where evaluation results will be located")
        args = parser.parse_args()

        img_root = args.img_root
        eval_out_dir = args.eval_out_dir

        if not os.path.exists(eval_out_dir):
            os.mkdir(eval_out_dir)

        path_fake = os.path.join(img_root, "fake/")
        path_real = os.path.join(img_root, "real/")
        print("Evaluating \'cars\' images located at %s" % img_root)

        eval_res = evaluate_generated_images("all", path_fake, path_real)

        df = pd.DataFrame(eval_res, index=["cars"])
        df.to_pickle(os.path.join(eval_out_dir, 'eval.pkl'))
        df.to_csv(os.path.join(eval_out_dir, 'eval.csv'))
    else:
        categories = {'02958343': 'cars', '03001627': 'chairs',
                      '02691156': 'airplanes', '04379243': 'tables',
                      '02828884': 'benches', '02933112': 'cabinets',
                      '04256520': 'sofa', '03636649': 'lamps',
                      '04530566': 'vessels'}

        parser = argparse.ArgumentParser(
            description='Generate Color for given mesh.'
        )

        parser.add_argument('config', type=str, help='Path to config file.')

        args = parser.parse_args()
        cfg = config.load_config(args.config, 'configs/default.yaml')
        base_path = cfg['test']['vis_dir']

        if cfg['data']['shapes_multiclass']:
            category_paths = glob.glob(os.path.join(base_path, '*'))
        else:
            category_paths = [base_path]

        for category_path in category_paths:
            cat_id = os.path.basename(category_path)
            category = categories.get(cat_id, cat_id)
            path1 = os.path.join(category_path, 'fake/')
            path2 = os.path.join(category_path, 'real/')
            print('Evaluating %s (%s)' % (category, category_path))

            evaluation = evaluate_generated_images('all', path1, path2)
            name = base_path

            df = pd.DataFrame(evaluation, index=[category])
            df.to_pickle(os.path.join(category_path, 'eval.pkl'))
            df.to_csv(os.path.join(category_path, 'eval.csv'))
    print('Evaluation finished')
