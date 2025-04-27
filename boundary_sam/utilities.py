import numpy as np
import cv2
from shapely.geometry import Polygon
from skimage import measure
from pycocotools import mask as maskUtils

def bounding_box(mask):
    """
    Compute the bounding box of a binary mask.
    
    Args:
    mask (np.array): 2D binary mask.
    
    Returns:
    tuple: (min_x, min_y, max_x, max_y) coordinates of the bounding box, or None if the mask is empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None  # Empty mask
    
    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]
    
    return (min_x, min_y, max_x, max_y)


def get_annotations_for_image(image_name, coco_data):
    image_id = None
    
    # Get image ID from image name
    for image in coco_data['images']:
        if image['file_name'] == image_name:
            image_id = image['id']
            break
    
    if image_id is None:
        return None

    # Get all annotations for the image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    return annotations

def generate_binary_masks(annotations, image_size):
    height, width = image_size  # Image dimensions: (height, width)
    
    # Initialize an empty list to store binary masks
    binary_masks = []

    for annotation in annotations:
        segmentation = annotation['segmentation']

        # Create an empty binary mask for the current image
        binary_mask = np.zeros((height, width), dtype=np.uint8)

        # If the segmentation is a polygon, use it to create a binary mask
        if isinstance(segmentation, list):  # Polygonal segmentation
            # Ensure the segmentation is in the format expected by pycocotools (list of lists)
            if all(isinstance(seg, list) for seg in segmentation):
                rle = maskUtils.frPyObjects(segmentation, height, width)
            else:
                # Wrap the segmentation in an outer list if it's not already a list of lists
                rle = maskUtils.frPyObjects([segmentation], height, width)
            
            binary_mask = maskUtils.decode(rle).astype(np.uint8)

        binary_masks.append(binary_mask)

    return binary_masks


def sam_mask_to_edges(mask):
    binary_mask = (mask > 0).astype(np.uint8)
    kernel = np.ones((2, 2), np.uint8)
    eroded_mask = cv2.erode(binary_mask, kernel, iterations=1)
    outer_edge = binary_mask - eroded_mask
    return outer_edge


def post_process(masks_w):
    masks_w = [x['segmentation']  for x in masks_w]
    sam_res = np.array(masks_w)
    sam_res1 = sam_res.sum(axis=0)

    edges = list(map(sam_mask_to_edges,masks_w))
    edges = np.array(edges)
    edges = np.sum(edges, axis = 0)
    edges1 = (edges>0).astype('uint8')

    seg = ((sam_res1)>0).astype('uint8')
    final_res = seg*(1 - ((  edges1)>0).astype('uint8'))
    final_res = (final_res>0).astype('uint8')
    return final_res



### metric functions

def bounding_box(mask):
    """
    Compute the bounding box of a binary mask.
    
    Args:
    mask (np.array): 2D binary mask.
    
    Returns:
    tuple: (min_x, min_y, max_x, max_y) coordinates of the bounding box, or None if the mask is empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None  # Empty mask
    
    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]
    
    return (min_x, min_y, max_x, max_y)

def calculate_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
    bb1, bb2 (tuple): Bounding boxes in the format (min_x, min_y, max_x, max_y).
    
    Returns:
    float: IoU value between 0 and 1.
    """
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No intersection

    # Area of intersection
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # Areas of the bounding boxes
    bb1_area = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    bb2_area = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    # Area of the union
    union_area = bb1_area + bb2_area - intersection_area

    return intersection_area / union_area


def generate_binary_masks(annotations, image_size):
    height, width = image_size  # Image dimensions: (height, width)
    
    # Initialize an empty list to store binary masks
    binary_masks = []

    for annotation in annotations:
        segmentation = annotation['segmentation']

        # Create an empty binary mask for the current image
        binary_mask = np.zeros((height, width), dtype=np.uint8)

        # If the segmentation is a polygon, use it to create a binary mask
        if isinstance(segmentation, list):  # Polygonal segmentation
            # Ensure the segmentation is in the format expected by pycocotools (list of lists)
            if all(isinstance(seg, list) for seg in segmentation):
                rle = maskUtils.frPyObjects(segmentation, height, width)
            else:
                # Wrap the segmentation in an outer list if it's not already a list of lists
                rle = maskUtils.frPyObjects([segmentation], height, width)
            
            binary_mask = maskUtils.decode(rle).astype(np.uint8)

        binary_masks.append(binary_mask)

    return binary_masks



def mask_to_polygon(mask):
    """
    Convert a binary mask to a polygon using the mask's contours.
    
    Args:
    mask (np.array): 2D binary mask.
    
    Returns:
    Polygon: Shapely polygon representing the mask's region.
    """
    contours = measure.find_contours(mask, 0.5)
    
    if len(contours) == 0:
        return None  # No contours, return None for empty mask
    
    # Convert the largest contour to a Polygon (assuming single connected region)
    largest_contour = max(contours, key=lambda x: x.shape[0])
    
    # Polygon coordinates are in (y, x), so we need to reverse them to (x, y)
    return Polygon([(x, y) for y, x in largest_contour])

def calculate_os_polygon(gt_polygon, pred_polygon):
    """
    Calculate Over-Segmentation (OS) for two polygons (ground truth and prediction).
    
    Args:
    gt_polygon (Polygon): Ground truth polygon.
    pred_polygon (Polygon): Prediction polygon.
    
    Returns:
    float: Over-segmentation metric.
    """
    if not gt_polygon or not pred_polygon:
        return 0

    intersection = gt_polygon.intersection(pred_polygon)
    pred_area = pred_polygon.area
    
    if pred_area == 0:
        return 0  # No prediction, no over-segmentation
    
    return 1 - (intersection.area / pred_area)

def calculate_us_polygon(gt_polygon, pred_polygon):
    """
    Calculate Under-Segmentation (US) for two polygons (ground truth and prediction).
    
    Args:
    gt_polygon (Polygon): Ground truth polygon.
    pred_polygon (Polygon): Prediction polygon.
    
    Returns:
    float: Under-segmentation metric.
    """
    if not gt_polygon or not pred_polygon:
        return 0

    intersection = gt_polygon.intersection(pred_polygon)
    gt_area = gt_polygon.area
    
    if gt_area == 0:
        return 0  # No ground truth, no under-segmentation
    
    return 1 - (intersection.area / gt_area)

def calculate_iou_polygon(gt_polygon, pred_polygon):
    """
    Calculate Intersection over Union (IoU) for two polygons (ground truth and prediction).
    
    Args:
    gt_polygon (Polygon): Ground truth polygon.
    pred_polygon (Polygon): Prediction polygon.
    
    Returns:
    float: Intersection over Union metric.
    """
    if not gt_polygon or not pred_polygon:
        return 0.0

    # Calculate the intersection and union of the polygons
    intersection = gt_polygon.intersection(pred_polygon)
    union = gt_polygon.union(pred_polygon)

    # Calculate areas
    intersection_area = intersection.area
    union_area = union.area

    if union_area == 0:
        return 0.0  # No union means no overlap, hence IoU is zero.

    # Calculate IoU
    return intersection_area / union_area


def filter_intersecting_bboxes_with_iou(gt_masks, pred_masks, iou_thresh=0.25):
    """
    Filter intersecting bounding boxes from two lists of masks (ground truth and prediction),
    and only keep those with IoU greater than the specified threshold.
    
    Args:
    gt_masks (list of np.array): List of ground truth binary masks.
    pred_masks (list of np.array): List of prediction binary masks.
    iou_thresh (float): IoU threshold for filtering, default is 0.1.
    
    Returns:
    dict: Dictionary where each GT bounding box index maps to a list of intersecting prediction bounding box indices.
    """
    gt_bboxes = [bounding_box(mask) for mask in gt_masks]
    pred_bboxes = [bounding_box(mask) for mask in pred_masks]

    intersections = {}

    for gt_idx, gt_bb in enumerate(gt_bboxes):
        if gt_bb is None:
            print("errrrr")
            break
            continue  # Skip empty ground truth masks
            
        
        intersecting_preds = []
        for pred_idx, pred_bb in enumerate(pred_bboxes):
            if pred_bb is None:
                continue  # Skip empty prediction masks
            
            iou = calculate_iou(gt_bb, pred_bb)
            if iou >= iou_thresh:
                intersecting_preds.append((pred_idx, iou))  # Store index and IoU
        
        intersections[gt_idx] = intersecting_preds

    return intersections

def calculate_metrics(masks_gt,masks_w,IoU_thresh = 0.25):
    intersections = filter_intersecting_bboxes_with_iou(masks_gt, masks_w,IoU_thresh)
    # Display the intersecting bounding boxes
    us_list = [-1 for i in range(len(masks_gt))]
    os_list = [-1 for i in range(len(masks_gt))]
    iou_list = [-1 for i in range(len(masks_gt))]

    for gt_idx, pred_indices in intersections.items():
        # print(f"GT Mask {gt_idx} intersects with prediction masks: {pred_indices}")
        
        us_per_gt = []
        os_per_gt = []
        iou_per_gt = []
        # print(len(pred_indices))
        
        for pred_indx in pred_indices:
            try:
                us_ = calculate_us_polygon(mask_to_polygon(masks_gt[gt_idx]),mask_to_polygon(masks_w[pred_indx[0]]))
                os_ = calculate_os_polygon(mask_to_polygon(masks_gt[gt_idx]),mask_to_polygon(masks_w[pred_indx[0]]))
                iou_ = calculate_iou_polygon(mask_to_polygon(masks_gt[gt_idx]),mask_to_polygon(masks_w[pred_indx[0]]))
                # print(us_,os_)
                us_per_gt.append(us_)
                os_per_gt.append(os_)
                iou_per_gt.append(iou_)
            except:
                us_per_gt = [-2]
                os_per_gt = [-2]
                iou_per_gt = [-2]
                
        
        if  len(pred_indices)>0:
            us_per_gt_final = np.asarray(us_per_gt).mean()
            os_per_gt_final = np.asarray(os_per_gt).mean()
            iou_per_gt_final = np.asarray(iou_per_gt).mean()
        else:
            us_per_gt_final = -1
            os_per_gt_final = -1
            iou_per_gt_final = -1
            

        us_list[gt_idx] = us_per_gt_final
        os_list[gt_idx] = os_per_gt_final
        iou_list[gt_idx] = iou_per_gt_final

    us_np = np.asarray(us_list)
    os_np = np.asarray(os_list)
    iou_np = np.asarray(iou_list)
    return np.mean(us_np[us_np > -1]), np.mean(os_np[os_np > -1]), np.sum(us_np == -1)/len(us_np),np.mean(iou_np[iou_np > -1])


