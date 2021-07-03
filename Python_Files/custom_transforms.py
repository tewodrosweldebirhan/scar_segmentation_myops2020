import PIL
import numpy as np
import random
import numbers
import torchvision.transforms.functional as FF

class RandomRotate(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    def __call__(self, sample):
        """
        Args:
            img(Ndarray): Image to be rotated.
            mask(Ndarray): mask to be rotated

        Returns:
            Ndarray Image: Rotated image and mask.
        """

        img, mask = sample['image'], sample['mask']        

        #Random angle between -degrees, degrees
        angle = random.uniform(self.degrees[0], self.degrees[1])

        #Convert ndarray to PIL image
        mask = FF.to_pil_image(np.uint8(mask[0]), 'L') #
        img0 = FF.to_pil_image(np.float32(img[0]), 'F')
        img1 = FF.to_pil_image(np.float32(img[1]), 'F')
        img2 = FF.to_pil_image(np.float32(img[2]), 'F')

        rot_mask = FF.rotate(mask, angle, PIL.Image.NEAREST)
        rot_image0 = FF.rotate(img0, angle, PIL.Image.BILINEAR)
        rot_image1 = FF.rotate(img1, angle, PIL.Image.BILINEAR)
        rot_image2 = FF.rotate(img2, angle, PIL.Image.BILINEAR)
        
        rot_image0, rot_image1, rot_image2 = np.array(rot_image0), np.array(rot_image1), np.array(rot_image2)  
        
        rot_image0, rot_image1, rot_image2 = np.expand_dims(rot_image0, axis=0), np.expand_dims(rot_image1, axis=0),np.expand_dims(rot_image2, axis=0)  
     
        
        rot_image = np.concatenate((rot_image0, rot_image1, rot_image2), axis=0)
        
        rot_mask = np.array(rot_mask)
        rot_mask = np.float32(np.expand_dims(rot_mask, axis=0))        


        return {'image':rot_image, 'mask':rot_mask}

class MyRandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be apllied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image And int for grayscale) for the area
            outside the transform in the output image.(Pillow>=5.0.0)

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and \
                    (len(shear) == 2 or len(shear) == 4), \
                    "shear should be a list or tuple and it must be of length 2 or 4."
                # X-Axis shear with [min, max]
                if len(shear) == 2:
                    self.shear = [shear[0], shear[1], 0., 0.]
                elif len(shear) == 4:
                    self.shear = [s for s in shear]
        else:
            self.shear = shear

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 2:
                shear = [random.uniform(shears[0], shears[1]), 0.]
            elif len(shears) == 4:
                shear = [random.uniform(shears[0], shears[1]),
                         random.uniform(shears[2], shears[3])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, sample):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        img, mask = sample['image'], sample['mask'] 

              

        #Random angle between -degrees, degrees
        angle = random.uniform(self.degrees[0], self.degrees[1])

        #Convert ndarray to PIL image
        mask = FF.to_pil_image(np.uint8(mask[0]), 'L') #
        img0 = FF.to_pil_image(np.float32(img[0]), 'F')
        img1 = FF.to_pil_image(np.float32(img[1]), 'F')
        img2 = FF.to_pil_image(np.float32(img[2]), 'F')
        
        
        #Parameters
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, mask.size) 

        rot_mask = FF.affine(mask, *ret, resample=PIL.Image.NEAREST)
        rot_image0 = FF.affine(img0, *ret, resample=PIL.Image.BILINEAR)
        rot_image1 = FF.affine(img1, *ret, resample=PIL.Image.BILINEAR)
        rot_image2 = FF.affine(img2, *ret, resample=PIL.Image.BILINEAR)
        
        
        rot_image0, rot_image1, rot_image2 = np.array(rot_image0), np.array(rot_image1), np.array(rot_image2)        
                
        
        rot_image0, rot_image1, rot_image2 = np.expand_dims(rot_image0, axis=0), np.expand_dims(rot_image1, axis=0),np.expand_dims(rot_image2, axis=0)  
        
        rot_image = np.concatenate((rot_image0, rot_image1, rot_image2), axis=0)
        
        rot_mask = np.array(rot_mask)
        rot_mask = np.float32(np.expand_dims(rot_mask, axis=0))        

        return {'image':rot_image, 'mask':rot_mask}
                
class RandomHorizFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img(Ndarray): Image to be flipped.
            mask(Ndarray): mask to be flipped

        Returns:
            Ndarray Image: Randomly flipped image.
        """

        img, mask = sample['image'], sample['mask']        

        #Convert ndarray to PIL image
        mask_ = FF.to_pil_image(np.uint8(mask[0]), 'L') # 8-bit uint
        img0 = FF.to_pil_image(np.float32(img[0]), 'F') #32 Floating point
        img1 = FF.to_pil_image(np.float32(img[1]), 'F')
        img2 = FF.to_pil_image(np.float32(img[2]), 'F')       
   
        if random.random() < self.p:

          rot_mask = FF.hflip(mask_)
          rot_image0 = FF.hflip(img0)
          rot_image1 = FF.hflip(img1)
          rot_image2 = FF.hflip(img2)

          rot_image0, rot_image1, rot_image2 = np.array(rot_image0), np.array(rot_image1), np.array(rot_image2)        
          rot_image0, rot_image1, rot_image2 = np.expand_dims(rot_image0, axis=0), np.expand_dims(rot_image1, axis=0),np.expand_dims(rot_image2, axis=0)  
          rot_image = np.concatenate((rot_image0, rot_image1, rot_image2), axis=0)
          
          rot_mask = np.array(rot_mask)
          rot_mask = np.expand_dims(rot_mask, axis=0)

          return {'image':rot_image, 'mask':rot_mask}


        return {'image':img, 'mask':mask}
   

class RandomVertFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img(Ndarray): Image to be flipped.
            mask(Ndarray): mask to be flipped

        Returns:
            Ndarray Image: Randomly flipped image.
        """

        img, mask = sample['image'], sample['mask']        

        #Convert ndarray to PIL image
        mask_ = FF.to_pil_image(np.uint8(mask[0]), 'L') # 8-bit uint
        img0 = FF.to_pil_image(np.float32(img[0]), 'F') #32 Floating point
        img1 = FF.to_pil_image(np.float32(img[1]), 'F')
        img2 = FF.to_pil_image(np.float32(img[2]), 'F')       
   
        if random.random() < self.p:

          rot_mask = FF.vflip(mask_)
          rot_image0 = FF.vflip(img0)
          rot_image1 = FF.vflip(img1)
          rot_image2 = FF.vflip(img2)

          rot_image0, rot_image1, rot_image2 = np.array(rot_image0), np.array(rot_image1), np.array(rot_image2)        
          rot_image0, rot_image1, rot_image2 = np.expand_dims(rot_image0, axis=0), np.expand_dims(rot_image1, axis=0),np.expand_dims(rot_image2, axis=0)  
          rot_image = np.concatenate((rot_image0, rot_image1, rot_image2), axis=0)
          
          rot_mask = np.array(rot_mask)
          rot_mask = np.expand_dims(rot_mask, axis=0)

          return {'image':rot_image, 'mask':rot_mask}


        return {'image':img, 'mask':mask}
