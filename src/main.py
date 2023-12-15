import argparse
import os
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from plotter import get_sigmoid_curve

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Curve Plotter', add_help=add_help)
    parser.add_argument('--data_path', default='./data', type=str, help='path of data')
    parser.add_argument('--x_axis', default='', type=str, help='data of x-axis')
    parser.add_argument('--y_axis', default='', type=str, help='data of y-axis')
    parser.add_argument('--x_extend', default=5, type=int, help='draw lower pp curve')
    parser.add_argument('--x_axis_spacing', default=10, type=int, help='space of x-axis')
    parser.add_argument('--y_axis_spacing', default=2000, type=int, help='space of y-axis')
    parser.add_argument('--output_dir', default='./result', type=str, help='path to save outputs')
    return parser
    
def main(args):
    max_x = 1
    max_y = 1
    
    plt.figure(figsize=(10.8, 5.4))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    file_path = os.path.join(args.data_path, "*.csv")
    for file in glob.glob(file_path):
        df = pd.read_csv(file)
        data_x = df[args.x_axis]
        data_y = df[args.y_axis]
        
        x, y = get_sigmoid_curve(data_x, data_y, args.x_extend)
        
        name = os.path.splitext(os.path.basename(file))[0]
        #plt.plot(data_x, data_y, 'o', label=name+'_data')
        plt.plot(x, y, label=name)
        
        if max(x) > max_x: max_x = max(x)
        if max(y) > max_y: max_y = max(y)

    plt.xlim(0, max_x*1.1)
    plt.ylim(0, max_y*1.1)
    plt.xticks(np.arange(0, max_x*1.1, args.x_axis_spacing))
    plt.yticks(np.arange(0, max_y*1.1, args.y_axis_spacing))
    plt.legend(loc='best')
    
    file_name = 'curve_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.png'
    path = os.path.join(args.output_dir, file_name)
    os.makedirs(args.output_dir, exist_ok=True)
    plt.savefig(path)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)