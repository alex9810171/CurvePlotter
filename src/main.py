import argparse
import os
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from plotter import get_sigmoid_curve, get_log_curve

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Curve Plotter', add_help=add_help)
    parser.add_argument('--data_path', default='./data', type=str, help='path of data')
    parser.add_argument('--title', default='Curve', type=str, help='Graph Title')
    parser.add_argument('--x_axis', default='', type=str, help='data of x-axis')
    parser.add_argument('--y_axis', default='', type=str, help='data of y-axis')
    parser.add_argument('--x_lower_bound', default=5, type=int, help='draw lower pp curve')
    parser.add_argument('--x_axis_spacing', default=10, type=int, help='space of x-axis')
    parser.add_argument('--y_axis_spacing', default=2000, type=int, help='space of y-axis')
    parser.add_argument('--hide_curve', dest='display_curve', default=True, action='store_false', help='whether to display curve, default is True')
    parser.add_argument('--display_dot', dest='display_dot', default=False, action='store_true', help='whether to display dot, default is False')
    parser.add_argument('--output_dir', default='./result', type=str, help='path to save outputs')
    return parser
    
def main(args):
    max_x = 1
    max_y = 1
    
    plt.figure(figsize=(10.8, 5.4))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.title(args.title, fontsize=20)
    plt.xlabel(args.x_axis, fontsize=16)
    plt.ylabel(args.y_axis, fontsize=16)
    
    file_path = os.path.join(args.data_path, "*.csv")
    for file in glob.glob(file_path):
        df = pd.read_csv(file)
        data_x = df[args.x_axis]
        data_y = df[args.y_axis]

        if len(data_x) > 3 and (max(data_x)-min(data_x)) > (2/3)*max(data_x):
            x, y = get_sigmoid_curve(data_x, data_y, args.x_lower_bound)
        else:
            x, y = get_log_curve(data_x, data_y, args.x_lower_bound)
            
        name = os.path.splitext(os.path.basename(file))[0]
        if args.display_curve:
            plt.plot(x, y, label=name)
        if args.display_dot:
            plt.plot(data_x, data_y, 'o', label=name+'_data')
        
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