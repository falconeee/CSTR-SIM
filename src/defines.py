from os.path import expanduser
import platform

system = platform.system()
homepath = expanduser('~')

if system == 'Linux' or system == 'Darwin': # macOS é identificado como 'Darwin'
    clouddir = homepath + '/Nextcloud/'
elif system == 'Windows':
    if homepath.endswith('thera'):
        homepath = 'E:'
    clouddir = homepath + '\\Nextcloud\\'
else:
    clouddir = homepath + '/Nextcloud/' # Fallback de segurança

rootdir = clouddir + 'book/'
softdir1 = '/home/thomas/Dropbox/posgrad/patternrecog/soft'

twocolwid = 6.5  # Width of graphic element, where two elements are in one row
twocolhei = 6.5  # Height of graphic element, where two elements are in one row

defines = {
    'system'             :   system,
    'rootdir'            :   rootdir,
    'includedirs'        :   (rootdir + 'soft/lib', rootdir + 'soft/linear', softdir1,),
    'graphics_dim_two'   :   (twocolwid, twocolhei)
}