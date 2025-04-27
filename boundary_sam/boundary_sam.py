import json
import numpy as np
from skimage import measure
from pycocotools import mask as maskUtils
from shapely.geometry import Polygon
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.restoration import denoise_bilateral
import cv2
from matplotlib.patches import Polygon as plt_polygon
from matplotlib import pyplot as plt


class BoundarySAM:
    def __init__(self, checkpoint, model_type, SAM_ARGS, device = 'cuda'):
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        self.mask_generator  = SamAutomaticMaskGenerator(model=sam,
                                                         points_per_side= SAM_ARGS['PPS'],
                                                         pred_iou_thresh=SAM_ARGS['IoUthresh'],
                                                         stability_score_thresh=SAM_ARGS['SST'],
                                                         crop_n_layers = 0,
                                                         min_mask_region_area = 10,
                                                         )

    def generate_masks_original(self,im):
        return self.mask_generator.generate(im)

    def generate_masks_enhanced(self,im):
        enhanced_image = self.enhance_image(im)
        return self.mask_generator.generate(enhanced_image)

    def enhance_image(self, im):
        im_width, im_hight, c = im.shape
        
        # this effects the guided filter and it depends on the size of the image
        # for the datasets we use, we tested for 128 and 256 and we use 2 and 4 respectively.
        # you can always just use PCA enhancemend wihtout the filters, it does most of the work and is non parametric. 
        if im_width == 128:
            r_selected = 2
        elif im_width == 256:
            r_selected = 4
            
            
        
        # set the image to get the embeddings
        self.mask_generator.predictor.set_image(im)
        
        #get the embeddings
        embeddings = self.mask_generator.predictor.get_image_embedding()
        embeddings_np = embeddings.squeeze().detach().cpu().numpy()
        
        ## apply PCA
        H, W = embeddings_np.shape[1], embeddings_np.shape[2]
        data_reshaped = embeddings_np.reshape(256, H * W).T 
        scaler = StandardScaler()
        data_standardized = scaler.fit_transform(data_reshaped)
        n_components = 1
        pca = PCA(n_components=n_components)
        data_reduced = pca.fit_transform(data_standardized)
        data_final = data_reduced.T.reshape(H, W) 
        data_final = (data_final - data_final.min())/(data_final.max() - data_final.min())
        data_final = (1*data_final).astype('float32')
        
        # Decompose the image using bilateral filter and get the detail image.
        w = 5
        sigma = [1.6, 2*np.std(data_final, ddof=1)]
        im_filtered_1 = denoise_bilateral(
            data_final, win_size=w, sigma_color=sigma[1], sigma_spatial=sigma[0]
        )
        diff_1 = data_final - im_filtered_1
        wave_1 = diff_1 
        wave_1 = cv2.resize(
            wave_1, (im_hight, im_width), interpolation=cv2.INTER_CUBIC
        )
        wave_1 = (wave_1 - wave_1.min())/(wave_1.max() - wave_1.min())
        data_final = cv2.resize(
            data_final, (im_hight, im_width), interpolation=cv2.INTER_CUBIC
        )

        # infuse the detail image into the image using guided filtering.
        radius = r_selected
        epsilon = 0.01 
        R_new = cv2.ximgproc.guidedFilter(
            (255*im[:,:,0]).astype('uint8'),
            (255*wave_1).astype('uint8'),
            radius,
            epsilon
        )
        G_new = cv2.ximgproc.guidedFilter(
            (255*im[:,:,1]).astype('uint8'),
            (255*wave_1).astype('uint8'),
            radius,
            epsilon
        )
        B_new = cv2.ximgproc.guidedFilter(
            (255*im[:,:,2]).astype('uint8'),
            (255*wave_1).astype('uint8'),
            radius,
            epsilon
        )

        # normalize
        # R_new = (R_new - R_new.min())/(R_new.max() - R_new.min())
        # G_new = (G_new - G_new.min())/(G_new.max() - G_new.min())
        # B_new = (B_new - B_new.min())/(B_new.max() - B_new.min())
        im_new = np.stack((R_new, G_new, B_new), axis=2)
       
        return im_new
