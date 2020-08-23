from glob 						import glob
from os.path 					import splitext, basename
import cv2
import numpy as np
import scipy.fftpack
def filter(img):
    rows = img.shape[0]
    cols = img.shape[1]

        # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(img, dtype="float") / 255)

        # Create Gaussian mask of sigma = 10
    M = 2*rows + 1
    N = 2*cols + 1
    sigma = 10  
    (X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2

        # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
    Hhigh = 1 - Hlow

        # Move origin of filters so that it's at the top left corner to
        # match with the input image
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())

        # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
    Ioutlow = np.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
    Iouthigh = np.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))

        # Set scaling factors and add
    gamma1 = 0.3
    gamma2 = 1.5
    Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows,0:cols]

        # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255*Ihmf, dtype="uint8")
    return Ihmf2

input_dir  = 'output_LP'
output_dir = 'img_after_filter'

imgs_paths = sorted(glob('%s/*.png' % input_dir))
for i,img_path in enumerate(imgs_paths):
    bname = basename(img_path)
    img = cv2.imread(img_path,0)
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = filter(img)
    print(img.shape)
    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # cv2.imwrite("%s/%s" %(output_dir,bname),img)