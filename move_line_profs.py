import glob
import numpy as np

import shutil


def move_line_profs(res_dir, out_dir, lines_in, lines_out):
    epoch_dirs = glob.glob(res_dir + 'epoch*')
    epochs = [ int(d[-3:]) for d in epoch_dirs ]

    for i, epoch in enumerate(epochs):
        for j in range(len(lines_in)):
            
            in_file = epoch_dirs[i] + r'/' + lines_in[j] + '_profile.csv'
            out_file = out_dir + lines_out[j] + '/' + 'epoch{:03d}.csv'.format(epoch)
        
            shutil.copyfile(in_file, out_file)
            
    return


if __name__ == '__main__':
    res_dir = '/data3/stone28/2drm/sdssrm/fit_res_hb/rm160/'
    out_dir = '/data3/stone28/2drm/sdssrm/line_profs2/rm160/'

    lines_in = ['Hb_br']
    lines_out = ['hb']

    move_line_profs(res_dir, out_dir, lines_in, lines_out)
