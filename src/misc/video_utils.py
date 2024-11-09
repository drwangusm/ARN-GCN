import math
import cv2


def resize_img(img, new_size, keep_ratio=True, keep_ratio_mode='crop'):
    """ Resize image into new size
    """

    if keep_ratio and keep_ratio_mode not in ['pad','crop']:
        raise Exception("Invalid keep_ratio_mode {}, choose between {}"
                        .format(keep_ratio_mode, ['pad','crop']))
    
    if not keep_ratio:
        new_img = cv2.resize(img, (new_size[1], new_size[0]))
    else:
        old_size = img.shape[:2] 
        target_size = new_size[0]
        
        if keep_ratio_mode == 'pad':
            ratio = float(target_size)/max(old_size)
        else:
            ratio = float(target_size)/min(old_size)
        
        temp_size = tuple( max(int(x*ratio),target_size) for x in old_size)
        new_img = cv2.resize(img, (temp_size[1], temp_size[0]))
        
        if keep_ratio_mode == 'pad':
            raise Exception("Functionality not implemented yet")
            # zeros_img = np.zeros((target_size, target_size))
            # zeros_img[off_h: off_h+target_size,
                          # off_w: off_w+target_size,:] = new_img
            # new_img = zeros_img
        else:
            off_h = (temp_size[0] - target_size)//2
            off_w = (temp_size[1] - target_size)//2
            new_img = new_img[off_h: off_h+target_size,
                                  off_w: off_w+target_size]
    
    return new_img

def crop_img(img, bbox, pad=0, new_size=(224,224)):
    """
    Crop the input image given a bounding box.
    Make the output a square by adding black pixels around the necessary border.
    Possibility to add a pad value that will enlarge the input bbox.

    Parameters
    ----------
    img : image matrix (Height, Width, ...)
    bbox : list of ints
        Bounding box with values: x, y, width, height.
    pad : int, optional
        Pad value to add to bbox. The default is 0.

    Returns
    -------
    Square image of cropped regions plus pad.

    """
    
    x, y, w, h = bbox
    
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    
    if pad > 0:
        x = max(0, x-pad)
        y = max(0, y-pad)
        w += 2*pad
        h += 2*pad
        
    crop_img = img[y:y+h, x:x+w]
    
    h, w = crop_img.shape[:2]
    if w > h:
        border = (w-h)/2
        pad_img = cv2.copyMakeBorder(crop_img, 
                          math.ceil(border), math.floor(border), 0, 0, 
                          cv2.BORDER_CONSTANT, value=0)
    elif h > w:
        border = (h-w)/2
        pad_img = cv2.copyMakeBorder(crop_img, 
                          0, 0, math.ceil(border), math.floor(border),
                          cv2.BORDER_CONSTANT, value=0)
    else:
        pad_img = crop_img
    
    if new_size is not None:
        pad_img = resize_img(pad_img, new_size)
    
    return pad_img
