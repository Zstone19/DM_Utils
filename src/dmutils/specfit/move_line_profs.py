import glob
import shutil


def move_line_profs(main_dir, rmid, line_name):
    
    rm_dir = main_dir + '/rm{:03d}/'.format(rmid)
    
    epoch_dirs = glob.glob(rm_dir + line_name + '/qsofit/epoch*')
    epochs = [ int(d[-3:]) for d in epoch_dirs ]
    
    for i in range(len(epochs)):
        in_file = epoch_dirs[i] + r'/raw_br_profile.csv'
        out_file = rm_dir + line_name + '/profile/epoch{:03d}.csv'.format(epochs[i])
        
        shutil.copyfile(in_file, out_file)
        
        
    return


if __name__ == '__main__':
    move_line_profs('/data3/stone28/2drm/sdssrm/', 160, 'mg2')
