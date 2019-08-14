# from .camvid import CamVid
from .cityscapes import Cityscapes
from .kitti import Kitti
# from .ritscapes import Ritscapes
from .imagenet import Imagenet
from .rotloader import Rotloader

__all__ = ['CamVid', 'Cityscapes','Ritscapes','Kitti','Imagenet','Rotloader']
