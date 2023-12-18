import argparse
import os
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from plotter import get_sigmoid_curve, get_log_curve

apple_color = ['darkgray', 'gray', 'slategray']
intel_color = ['dodgerblue', 'deepskyblue', 'cornflowerblue']
amd_color = ['red', 'darkred', 'tomato']
qual_color = ['blue', 'darkblue', 'royalblue']

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
    apple_color_count = 0
    intel_color_count = 0
    amd_color_count = 0
    qual_color_count = 0
    
    plt.figure(figsize=(10.8, 5.4))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.title(args.title, fontsize=20)
    plt.xlabel(args.x_axis, fontsize=16)
    plt.ylabel(args.y_axis, fontsize=16)
    
    file_path = os.path.join(args.data_path, "*.csv")
    file_list = glob.glob(file_path)
    file_list.sort()
    for file in file_list:
        df = pd.read_csv(file)
        x_y = pd.DataFrame(df, columns=[args.x_axis, args.y_axis])
        x_y = x_y.dropna(axis=0, how='any')
        data_x = x_y[args.x_axis]
        data_y = x_y[args.y_axis]

        if len(data_x) > 3 and (max(data_x)-min(data_x)) > (2/3)*max(data_x):
            x, y = get_sigmoid_curve(data_x, data_y, args.x_lower_bound)
        elif len(data_x) == 3:
            x, y = get_log_curve(data_x, data_y, args.x_lower_bound)
            
        name = os.path.splitext(os.path.basename(file))[0]
        specific_color = None
        if 'Apple' in file:
            specific_color = apple_color[apple_color_count % len(apple_color)]
            apple_color_count += 1
        elif 'Intel' in file:
            specific_color = intel_color[intel_color_count % len(intel_color)]
            intel_color_count += 1
        elif 'AMD' in file:
            specific_color = amd_color[amd_color_count % len(amd_color)]
            amd_color_count += 1
        elif 'Qualcomm' in file:
            specific_color = qual_color[qual_color_count % len(qual_color)]
            qual_color_count += 1
        
        if args.display_curve and len(data_x) >= 3:
            plt.plot(x, y, label=name, color=specific_color)
        if args.display_dot or len(data_x) < 3:
            plt.plot(data_x, data_y, 'o', label=name, color=specific_color)
        
        if max(data_x) > max_x: max_x = max(data_x)
        if max(data_y) > max_y: max_y = max(data_y)

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