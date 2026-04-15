from os.path import expanduser
import platform


system = platform.system()
homepath = expanduser('~')

if system == 'Linux':
    clouddir = homepath + '/Nextcloud/'
if system == 'Windows':
    homepath = expanduser('~')

    if homepath.endswith('thera'):
        homepath = 'E:'

    clouddir = homepath + '\\Nextcloud\\'

rootdir = clouddir + 'book/'
softdir1 = '/home/thomas/Dropbox/posgrad/patternrecog/soft'

twocolwid = 6.5	 # Width of graphic element, where two elements are in one row
twocolhei = 6.5	 # Height of graphic element, where two elements are in one row


defines = {
    'system'             :   system,
    'rootdir'            :   rootdir,
    'includedirs'        :   (rootdir + 'soft/lib', rootdir + 'soft/linear', softdir1,),    # observe the order
    'graphics_dim_two'   :   (twocolwid, twocolhei)
}
