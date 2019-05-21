'''
This function is referred to:
  http://awsgit.viscovery.co/Cybertron/scutils/blob/master/scutils/scdraw.py
'''

import sys
import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import cv2
import numpy as np
import random
import colorsys

import hashlib
import binascii

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def convert_contours2bbox(contours, axis=1):
    assert isinstance(
        contours, np.ndarray), 'type(contours) %s != np.ndarray' % type(contours)
    if len(contours.shape) == 3:  # (#points, 1, 2)
        contours = contours[np.newaxis, :]  # (#contours=1, #points, 1, 2)
    # upper left corner (x, y format)
    bbox_ul = np.min(contours, axis=axis)[0][0].tolist()
    # lower right corner (x, y format)
    bbox_lr = np.max(contours, axis=axis)[0][0].tolist()
    # x, y, w, h
    return [bbox_ul[0], bbox_ul[1], bbox_lr[0]-bbox_ul[0], bbox_lr[1]-bbox_ul[1]]


def random_color():
    h = random.random()
    s = 0.5 + random.random() / 2.0
    l = 0.4 + random.random() / 5.0
    r, g, b = (int(256 * i) for i in colorsys.hls_to_rgb(h, l, s))
    return (r, g, b)

def color_per_category(s):
    s = s.encode('utf-8')
    large_number = int(binascii.hexlify(s), 16) % (10 ** 12) 
    r = int(large_number/(10 ** 3)) % 256
    g = int(large_number/(10 ** 6)) % 256
    b = int(large_number/(10 ** 9)) % 256
    return (r, g, b)


def mkdir_if_missing(path):
    if not os.path.exists(path):
        os.makedirs(path)

def draw_textbox(img, pos, class_str,
                 font_url='https://github.com/googlei18n/noto-cjk/raw/master/NotoSerifTC-ExtraLight.otf',
                 font_sz=30,
                 bg_color=(18, 127, 15),
                 bg_alpha=0.4,
                 return_txt_bbox=False):
    # load font
    font_fn = font_url.split('/')[-1]
    long_font_fn = os.path.join(*[project_dir, 'data/fonts', font_fn])
    if not os.path.isfile(long_font_fn):  # if doesn't exist, download from web
        mkdir_if_missing(os.path.join(project_dir, 'data/fonts'))
        urlretrieve(font_url, long_font_fn)
    font = ImageFont.truetype(long_font_fn, font_sz)
    # init
    _img = img.copy()
    x0, y0 = int(pos[0]), int(pos[1])  # lower left corner
    # get textbox size
    txt_w, txt_h = font.getsize(class_str)
    # place text background
    back_tl = x0, y0 - int(txt_h)
    back_br = x0 + txt_w, y0
    bin_mask = np.zeros(_img.shape[:2])
    bin_mask[back_tl[1]:back_br[1], back_tl[0]:back_br[0]] = 255
    idx = np.nonzero(bin_mask)
    _img = _img.astype(np.float)  # convert to float for alpha blending
    _img[idx[0], idx[1], :] *= 1.0 - bg_alpha
    _img[idx[0], idx[1], :] += bg_alpha * np.array(bg_color)
    _img = _img.astype(np.uint8)  # convert back to uint8
    # decide text color by mean background color
    _mean_color = np.mean(_img[idx[0], idx[1], :])
    txt_color = (0, 0, 0) if _mean_color >= 125 else (255, 255, 255)
    # put text
    _img_pil = Image.fromarray(_img)  # convert from opencv to pil
    draw = ImageDraw.Draw(_img_pil)
    txt_tl = (x0, y0 - int(txt_h))
    draw.text(txt_tl, class_str, txt_color, font=font)
    _img = np.asarray(_img_pil)  # convert from pil to opencv

    if return_txt_bbox:
        # (x, y, w, h) of lower left corner
        return _img, [x0, y0, txt_w, txt_h]
    else:
        return _img


def draw_bbox(img, bbox, thick=1, bbox_color=(18, 127, 15)):
    """Draw a bbox on image

    Args:
        img: OpenCV image format
        bbox: in the format of (x, y, w, h)
    """
    _img = img.copy()
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(_img, (x0, y0), (x1, y1), bbox_color, thickness=thick)
    return _img


def draw_mask(img, bin_mask,
              fill=True, fill_color=(218, 227, 218), alpha=0.4,
              show_border=True, border_color=(255, 255, 255), border_thick=1):
    """Draw a binary mask on image

    Args:
        img: OpenCV format
        bin_mask: binary mask with 0s on background pixels
    """
    _img = img.copy()
    _img = _img.astype(np.float32)

    if fill:
        idx = np.nonzero(bin_mask)
        _img[idx[0], idx[1], :] *= 1.0 - alpha
        _img[idx[0], idx[1], :] += alpha * np.array(fill_color)

    if show_border:
        _, contours, _ = cv2.findContours(
            bin_mask.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(_img, contours, -1, border_color,
                         border_thick, cv2.LINE_AA)

    return _img.astype(np.uint8)


def draw_mask_from_contour(img, contour,
                           fill=True, fill_color=(218, 227, 218), alpha=0.4,
                           show_border=True, border_color=(255, 255, 255), border_thick=1):
    """Draw a binary mask on image

    Args:
        img: OpenCV format
        contour: OpenCV contour format  (1, #points, 1, 2)
    """
    _img = img.copy()
    _img = _img.astype(np.float32)

    if fill:
        mask = np.zeros((img.shape[:2]))
        contour = contour.astype(np.int32)  # force contour to be integer
        cv2.fillPoly(mask, pts=contour, color=(255))  # get mask from contour
        idx = np.nonzero(mask)
        _img[idx[0], idx[1], :] *= 1.0 - alpha
        _img[idx[0], idx[1], :] += alpha * np.array(fill_color)

    if show_border:
        cv2.drawContours(_img, contour, -1, border_color,
                         border_thick, cv2.LINE_AA)

    return _img.astype(np.uint8)


def vis_one_image(img,
                  categories=None,
                  bboxes=None,
                  contours=None,
                  font_sz=30,
                  border_offset=30):
    # verify that all data size are the same
    """Visualize single image with chinse text and bounding box.
        NOTE: This function would not execute assertion and check.
    """
    '''
    if categories and bboxes:
        assert len(categories) == len(bboxes), 'len(categories) %d != len(bboxes) %d' % (
            len(categories), len(bboxes))
    if categories and contours:
        assert len(categories) == len(contours), 'len(categories) %d != len(contours) %d' % (
            len(categories), len(contours))
    if bboxes and contours:
        assert len(bboxes) == len(contours), 'len(bboxes) %d != len(contours) %d' % (
            len(bboxes), len(contours))
    '''
    # init
    num_annos = len(bboxes) if bboxes else len(
        categories) if categories else len(contours) if contours else 0
    # init classification textbox pos
    cate_pos = (border_offset, img.shape[0]-border_offset)
    # fill a list of [None, None, ..] if it is None
    categories = [None for x in range(
        num_annos)] if not categories else categories
    bboxes = [None for x in range(num_annos)] if not bboxes else bboxes
    contours = [None for x in range(num_annos)] if not contours else contours

    # loop over each annotation
    for k in range(num_annos):
        # classification task, put textboxes on the lower left corner
        if not bboxes[k] and contours[k] is None:
            img, txt_bbox = draw_textbox(img, cate_pos, categories[k],
                                         font_sz=font_sz,
                                         bg_color=random_color(),
                                         return_txt_bbox=True)
            # update textbox lower left corner
            cate_pos = (cate_pos[0], cate_pos[1]-txt_bbox[3])
        # detection task
        else:
            color = color_per_category(categories[k])
            #color = random_color()  # init random colors (textbox, bbox, mask will use the same color)
            # plot contours
            if contours[k] is not None:
                img = draw_mask_from_contour(img, contours[k],
                                             fill_color=color,
                                             border_color=color,
                                             border_thick=3)
            # plot bbox
            if bboxes[k]:  # if bbox is explicitly specified, do nothing
                pass
            else:  # get bboxes from contour
                bboxes[k] = convert_contours2bbox(contours[k])
            img = draw_bbox(img, bboxes[k],
                            thick=3, bbox_color=color)
            # plot text
            if categories[k]:
                x, y, w, h = bboxes[k]
                bbox_coors = [(x, y), (x, y+h), (x+w, y+h),
                              (x, y+h), (x, y)]  # 4 corners
                # pick the next corner if textbox will be drawn outside canvas
                for j, bbox_coor in enumerate(bbox_coors):
                    _img, (tx, ty, tw, th) = draw_textbox(img, bbox_coor, categories[k],
                                                          font_sz=font_sz,
                                                          bg_color=color,
                                                          return_txt_bbox=True)
                    if j == 4:
                        break
                    if tx > border_offset and ty > border_offset \
                       and tx+tw < img.shape[1]-border_offset \
                       and ty+th < img.shape[0]-border_offset:
                        break
                img = _img  # update image

    return img

def vis_square(data, padsize=1, padval=0):
    """take an array of shape (n, height, width) or (n, height, width, channels)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize),
               (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=(padval, padval))

    # tile the filters into an image
    data = data.reshape(
        (n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    return data