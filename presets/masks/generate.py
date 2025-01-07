import torch
import numpy as np
import torchvision
from PIL import Image


class MaskGenerator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = self._random_sq_bbox(img,
                              mask_shape=(mask_h, mask_w),
                              image_size=self.image_size,
                              margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask
    
    def _random_sq_bbox(self, img, mask_shape, image_size=256, margin=(16, 16)):
        """Generate a random sqaure mask for inpainting
        """
        B, C, H, W = img.shape
        h, w = mask_shape
        margin_height, margin_width = margin
        maxt = image_size - margin_height - h
        maxl = image_size - margin_width - w

        # bb
        # t = np.random.randint(margin_height, maxt)
        # l = np.random.randint(margin_width, maxl)
        t = (margin_height + maxt) // 2
        l = (margin_width + maxl) // 2

        # make mask
        mask = torch.ones([B, C, H, W], device=img.device)
        mask[..., t:t+h, l:l+w] = 0

        return mask, t, t+h, l, l+w


if __name__ == "__main__":
    img_size = (1, 3, 256, 256)
    
    random_mask = MaskGenerator(mask_type='random', mask_prob_range=(0.7, 0.7))(torch.zeros(img_size))[0]
    torchvision.utils.save_image(random_mask, 'random_mask.png')
    
    box_mask = MaskGenerator(mask_type='box', mask_len_range=(128, 129))(torch.zeros(img_size))[0]
    torchvision.utils.save_image(box_mask, 'box_mask.png')
    
    
    # load_mask = (torchvision.io.read_image('random_mask.png') / 255).round().int()