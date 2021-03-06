import sys, os
import keras
import cv2
import traceback
import numpy as np

import darknet.python.darknet as dn

from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes,dknet_label_conversion
from darknet.python.darknet     import detect
from src.utils 				    import nms
from src.drawing_utils			import draw_label, draw_losangle, write2img
from src.label 					import lread, Label, readShapes


def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))
		
input_dir  = 'test'
output_dir = 'output_LP'
result_dir = 'output'

lp_threshold = .5
ocr_threshold = .4

ocr_weights = 'data/ocr/ocr-net.weights'
ocr_netcfg  = 'data/ocr/ocr-net.cfg'
ocr_dataset = 'data/ocr/ocr-net.data'
wpod_net_path = "data/lp-detector/wpod-net_update1"
wpod_net = load_model(wpod_net_path)
ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
ocr_meta = dn.load_meta(ocr_dataset)

imgs_paths = glob('%s/*.jpg' % input_dir)

print 'Searching for license plates using WPOD-NET'

for i,img_path in enumerate(imgs_paths):

    print '\t Processing %s' % img_path

    bname = splitext(basename(img_path))[0]
    Ivehicle = cv2.imread(img_path)
    h = Ivehicle.shape[0]
    w = Ivehicle.shape[1]
    ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
    side  = int(ratio*288.)
    bound_dim = min(side + (side%(2**4)),608)
    print "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio)

    Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
    

    if len(LlpImgs):
        Ilp = LlpImgs[0]
        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
        
        s = Shape(Llp[0].pts)
        x0 = int (Llp[0].pts[0][0]*w)
        y0 = int (Llp[0].pts[1][0]*h)
        x2 = int (Llp[0].pts[0][2]*w)
        y2 = int (Llp[0].pts[1][2]*h)
        x1 = int (Llp[0].pts[0][1]*w)
        y1 = int (Llp[0].pts[1][1]*h)
        x3 = int (Llp[0].pts[0][3]*w)
        y3 = int (Llp[0].pts[1][3]*h)
        lp = np.array([[x0,y0],[x1,y1],[x2,y2],[x3,y3]],np.int32)
        cv2.polylines(Ivehicle, [lp], True, (0,0,255), thickness=4)
        #lp_V = np.array([[(x0+X3)/2,(y0+Y3)/2],[(x1+x2)/2,(y1+y3)/2],[x2,y2],[x3,y3]],np.int32)
        # cv2.circle(Ivehicle,(x,y),30,(0,0,255),thickness=-1)
        # cv2.circle(Ivehicle,(x_w,y_h),30,(0,0,255),thickness=-1)
        # cv2.rectangle(Ivehicle,(x,y),(x_w,y_h),(255,0,0),thickness=3)
        #cv2.imshow('LP',Ilp)
        cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
        # # writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])
        R,(width,height) = detect(ocr_net, ocr_meta, '%s/%s_lp.png' % (output_dir,bname),thresh=ocr_threshold, nms=None)

        if len(R):

            L = dknet_label_conversion(R,width,height)
            L = nms(L,.45)
            L.sort(key=lambda x: (x.tl()[1]>0.38, x.tl()[0]))
            lp_str = ''.join([chr(l.cl()) for l in L])
            cv2.putText(Ivehicle,lp_str,(np.min([x0,x1,x2,x3]), np.min([y0,y1,y2,y3])-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255),thickness=3, lineType=cv2.LINE_AA)
            cv2.imwrite('%s/%s.jpg' % (result_dir,bname),Ivehicle)
        else:

            print 'No characters found'



