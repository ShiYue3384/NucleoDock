from tkinter import N
from ..utils.utils import get_abs_path
import os


def get_image_json_fps(dir_path, num_files=None):
    fps = []
    if num_files <= 0:
        num_files = float('inf')

    _fns = os.listdir(get_abs_path(dir_path))
    _fns = [fn for fn in _fns if (fn.startswith('dir_') and int(fn[4:-5]) < num_files)]
    _fps = [get_abs_path(dir_path, fn) for fn in _fns]
    fps += _fps
    return fps


def get_wiki_ids_fps(dir_path, num_files=None):
    fps = []
    _fns = os.listdir(get_abs_path(dir_path))
    if num_files > 0:
        _fns = _fns[:num_files]
    _fps = [get_abs_path(dir_path, fn) for fn in _fns]
    fps += _fps
    return fps
