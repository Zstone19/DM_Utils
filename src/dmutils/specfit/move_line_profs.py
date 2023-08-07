import glob
import shutil


def move_line_profs(main_dir, rmid, line_name):
    
    if line_name == 'ha':
        line_in = 'Ha_br'
    elif line_name == 'hb':
        line_in = 'Hb_br'
    elif line_name == 'mg2':
        line_in = 'MgII_br'
    
    
    rm_dir = main_dir + '/rm{:03d}/'.format(rmid)
    
    epoch_dirs = glob.glob(rm_dir + line_name + '/epoch*')
    epochs = [ int(d[-3:]) for d in epoch_dirs ]
    
    for i in range(len(epochs)):
        in_file = epoch_dirs[i] + r'/' + line_in + '_profile.csv'
        out_file = rm_dir + line_name + '/profile/epoch{:03d}.csv'.format(epochs[i])
        
        shutil.copyfile(in_file, out_file)
        
        
    return


if __name__ == '__main__':
    move_line_profs('/data3/stone28/2drm/sdssrm/', 160, 'mg2')
