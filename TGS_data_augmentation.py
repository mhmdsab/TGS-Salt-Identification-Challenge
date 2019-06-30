import cv2
import numpy as np
from scipy import ndimage
import math


def hflip(img,mask):

    hflipped_img = cv2.flip(img,1)
    hflipped_mask = cv2.flip(mask,1)

    return hflipped_img,hflipped_mask


def vflip(img,mask):

    vflipped_img = cv2.flip(img,0)
    vflipped_mask = cv2.flip(mask,0)

    return vflipped_img,vflipped_mask


def hvflip(img,mask):

    hvflipped_img = cv2.flip(img,-1)
    hvflipped_mask = cv2.flip(mask,-1)

    return hvflipped_img,hvflipped_mask


def imrotate(img,mask,angle):

    rotated_img = ndimage.rotate(img,angle,order = 0)
    rotated_mask = ndimage.rotate(mask,angle,order = 0)
            
    return rotated_img,rotated_mask


def transpose(img,mask):

    transposed_img = img.transpose()
    transposed_mask = mask.transpose()
            
    return transposed_img,transposed_mask



def rescaled_crops(img,mask,size):
    
    p1 = np.random.randint(low = 0,high = size/4)
    p2 = np.random.randint(low = 3*size/4,high = size)

    resized_img = cv2.resize(img,(size,size))
    cropped_img = resized_img[p1:p2,p1:p2]

    resized_mask = cv2.resize(mask,(size,size))
    cropped_mask = resized_mask[p1:p2,p1:p2]
        
    return cropped_img,cropped_mask


def compute_center_pad(H, W, factor=32):
    if H % factor == 0:
        dy0, dy1 = 0, 0
    else:
        dy = factor - H % factor
        dy0 = dy // 2
        dy1 = dy - dy0

    if W % factor == 0:
        dx0, dx1 = 0, 0
    else:
        dx = factor - W % factor
        dx0 = dx // 2
        dx1 = dx - dx0

    return dy0, dy1, dx0, dx1


def do_center_pad_to_factor(image, factor=32):
    H, W = image.shape[:2]
    dy0, dy1, dx0, dx1 = compute_center_pad(H, W, factor)

    image = cv2.copyMakeBorder(image, dy0, dy1, dx0, dx1, cv2.BORDER_REFLECT_101)
    # cv2.BORDER_CONSTANT, 0)
    return image


def do_center_pad_to_factor2(image, mask, factor=32):
    image = do_center_pad_to_factor(image, factor)
    mask = do_center_pad_to_factor(mask, factor)
    return image, mask



def do_invert_intensity(image):
    # flip left-right
    image = np.clip(1 - image, 0, 1)
    return image


def do_brightness_shift(image, alpha=0.125):
    image = image + alpha
    image = np.clip(image, 0, 1)
    return image


def do_brightness_multiply(image, alpha=1):
    image = alpha * image
    image = np.clip(image, 0, 1)
    return image


# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def do_gamma(image, gamma=1.0):
    image = image ** (1.0 / gamma)
    image = np.clip(image, 0, 1)
    return image


def do_shift_scale_rotate2(image, mask, dx=2, dy=2, scale=1.3, angle=45):
    borderMode = cv2.BORDER_REFLECT_101
    # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    sx = scale
    sy = scale
    cc = math.cos(angle / 180 * math.pi) * (sx)
    ss = math.sin(angle / 180 * math.pi) * (sy)
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=borderMode, borderValue=(
        0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,  # cv2.INTER_LINEAR
                               borderMode=borderMode, borderValue=(
        0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = (mask > 0.5).astype(np.float32)
    return image, mask

# https://www.kaggle.com/ori226/data-augmentation-with-elastic-deformations
# https://github.com/letmaik/lensfunpy/blob/master/lensfunpy/util.py
def do_elastic_transform2(image, mask, grid=32, distort=0.2):
    borderMode = cv2.BORDER_REFLECT_101
    height, width = image.shape[:2]

    x_step = int(grid)
    xx = np.zeros(width, np.float32)
    prev = 0
    for x in range(0, width, x_step):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * (1 + np.random.uniform(-distort, distort))

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = int(grid)
    yy = np.zeros(height, np.float32)
    prev = 0
    for y in range(0, height, y_step):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * (1 + np.random.uniform(-distort, distort))

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    # grid
    map_x, map_y = np.meshgrid(xx, yy)
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)

    # image = map_coordinates(image, coords, order=1, mode='reflect').reshape(shape)
    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=borderMode,
                      borderValue=(0, 0, 0,))

    mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=borderMode, borderValue=(0, 0, 0,))
    mask = (mask > 0.5).astype(np.float32)
    return image, mask


def do_horizontal_shear2(image, mask, dx=.3):
    borderMode = cv2.BORDER_REFLECT_101
    # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    height, width = image.shape[:2]
    dx = int(dx * width)

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ], np.float32)
    box1 = np.array([[+dx, 0], [width + dx, 0], [width - dx, height], [-dx, height], ], np.float32)

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)

    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                borderMode=borderMode, borderValue=(
        0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST,  # cv2.INTER_LINEAR
                               borderMode=borderMode, borderValue=(
        0, 0, 0,))  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    mask = (mask > 0.5).astype(np.float32)
    return image, mask

#
#
#original_path = 'C:/Users/User/Desktop/original'
#mask_path = 'C:/Users/User/Desktop/mask'
#original_dst = 'C:/Users/User/Desktop/train_original'
#mask_dst = 'C:/Users/User/Desktop/train_mask'
#
#original_list = os.listdir(original_path)
#original_list.remove('Thumbs.db')
#mask_list = os.listdir(mask_path)
#mask_list.remove('Thumbs.db')
#
#
#
#for i in range(len(original_list)):
#    original_img = cv2.imread(os.path.join(original_path,original_list[i]),1)
#    mask = cv2.imread(os.path.join(mask_path,mask_list[i]),0)
#    hflipped_original,hflipped_mask = hflip(original_img,mask)
#    cv2.imwrite(os.path.join(original_dst,'hflipped'+original_list[i]),hflipped_original)
#    cv2.imwrite(os.path.join(mask_dst,'hflipped'+mask_list[i]),hflipped_mask)
#
#
#for i in range(len(original_list)):
#    original_img = cv2.imread(os.path.join(original_path,original_list[i]),1)
#    mask = cv2.imread(os.path.join(mask_path,mask_list[i]),0)
#    vflipped_original,vflipped_mask = vflip(original_img,mask)
#    cv2.imwrite(os.path.join(original_dst,'vflipped'+original_list[i]),vflipped_original)
#    cv2.imwrite(os.path.join(mask_dst,'vflipped'+mask_list[i]),vflipped_mask)
#    
#    
#for i in range(len(original_list)):
#    original_img = cv2.imread(os.path.join(original_path,original_list[i]),1)
#    mask = cv2.imread(os.path.join(mask_path,mask_list[i]),0)
#    hvflipped_original,hvflipped_mask = hvflip(original_img,mask)
#    cv2.imwrite(os.path.join(original_dst,'hvflipped'+original_list[i]),hvflipped_original)
#    cv2.imwrite(os.path.join(mask_dst,'hvflipped'+mask_list[i]),hvflipped_mask)
#    
#
#for i in range(len(original_list)):
#    original_img = cv2.imread(os.path.join(original_path,original_list[i]),1)
#    mask = cv2.imread(os.path.join(mask_path,mask_list[i]),0)
#    rotated_list_original,rotated_list_mask = imrotate(original_img,mask)
#    for j in range(len(rotated_list_original)):
#        cv2.imwrite(os.path.join(original_dst,'rotated_'+str(j)+original_list[i]),rotated_list_original[j])
#        cv2.imwrite(os.path.join(mask_dst,'rotated_'+str(j)+mask_list[i]),rotated_list_mask[j])
#
#
#for i in range(len(original_list)):
#    original_img = cv2.imread(os.path.join(original_path,original_list[i]),1)
#    mask = cv2.imread(os.path.join(mask_path,mask_list[i]),0)
#    zoomed_list_original,zoomed_list_mask = zoom(original_img,mask)
#    for j in range(len(zoomed_list_original)):
#        cv2.imwrite(os.path.join(original_dst,'zoomed_'+str(j)+original_list[i]),zoomed_list_original[j])
#        cv2.imwrite(os.path.join(mask_dst,'zoomed_'+str(j)+mask_list[i]),zoomed_list_mask[j])
#
#
#for i in range(len(original_list)):
#    original_img = cv2.imread(os.path.join(original_path,original_list[i]),1)
#    mask = cv2.imread(os.path.join(mask_path,mask_list[i]),0)
#    hvflipped_zoomed_list_original,hvflipped_zoomed_list_mask = hv_flip_zoomed(original_img,mask)
#    for j in range(len(hvflipped_zoomed_list_original)):
#        cv2.imwrite(os.path.join(original_dst,'hvflipped_zoomed_'+str(j)+original_list[i]),hvflipped_zoomed_list_original[j])
#        cv2.imwrite(os.path.join(mask_dst,'hvflipped_zoomed_'+str(j)+mask_list[i]),hvflipped_zoomed_list_mask[j])
#
#
#for i in range(len(original_list)):
#    original_img = cv2.imread(os.path.join(original_path,original_list[i]),1)
#    mask = cv2.imread(os.path.join(mask_path,mask_list[i]),0)
#    hvflipped_rotated_list_original,hvflipped_rotated_list_mask = hv_flip_rotated(original_img,mask)
#    for j in range(len(hvflipped_rotated_list_original)):
#        cv2.imwrite(os.path.join(original_dst,'hvflipped_rotated_'+str(j)+original_list[i]),hvflipped_rotated_list_original[j])
#        cv2.imwrite(os.path.join(mask_dst,'hvflipped_rotated_'+str(j)+mask_list[i]),hvflipped_rotated_list_mask[j])
#
#
#for i in range(len(original_list)):
#    original_img = cv2.imread(os.path.join(original_path,original_list[i]),1)
#    mask = cv2.imread(os.path.join(mask_path,mask_list[i]),0)
#    hvflipped_zoomed_rotated_list_original,hvflipped_zoomed_rotated_list_mask = hvflip_zoomed_rotated(original_img,mask)
#    for j in range(len(hvflipped_zoomed_rotated_list_original)):
#        cv2.imwrite(os.path.join(original_dst,'hvflipped_zoomed_rotated_'+str(j)+original_list[i]),hvflipped_zoomed_rotated_list_original[j])
#        cv2.imwrite(os.path.join(mask_dst,'hvflipped_zoomed_rotated_'+str(j)+mask_list[i]),hvflipped_zoomed_rotated_list_mask[j])
#
#
#
#
#
#
#
#
#
#


#------------------------------------------------------testing-------------------------------------------------------------------
##
##image_list = os.listdir('E:\Mohammed\IT\machine learning course\ml projects\TGS Salt Identification Challenge\Data\Train\images')
##mask_list = os.listdir('E:\Mohammed\IT\machine learning course\ml projects\TGS Salt Identification Challenge\Data\Train\masks')
##
##a,b = hvflip_zoomed_rotated('C:/Users/User/Desktop/581028_3804423466887_1552200814_n.jpg','C:/Users/User/Desktop/ahm.jpg')
##for i in range(len(a)):
##    cv2.imshow('zoom_a_'+str(i),a[i])
##    cv2.imshow('zoom_b_'+str(i),b[i])
##    cv2.waitKey(100)
##

##path = 'C:/Users/User/Desktop/original'
##dst = 'C:/Users/User/Desktop/mask'
##image_list = os.listdir(path)
##for item in image_list:
##    if not '.db' in item:
##        img = cv2.imread(os.path.join(path,item),0)
##        cv2.imwrite(os.path.join(dst,item),img)
##        cv2.waitKey(100)
##        







