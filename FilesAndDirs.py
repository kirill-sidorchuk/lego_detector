import os


def clear_directory(dir_name):
    files = os.listdir(dir_name)
    for file_name in files:
        path_to_remove = os.path.join(dir_name, file_name)
        if os.path.isdir(path_to_remove):
            clear_directory(path_to_remove)
            os.rmdir(path_to_remove)
        else:
            os.remove(path_to_remove)


def get_image_names_from_dir(data_dir):
    _files = os.listdir(data_dir)
    image_files = []
    for f in _files:
        if f.lower().endswith('.jpg'):
            image_files.append(os.path.join(data_dir, f))
    return image_files


def get_downsampled_img_name(img_file):
    ds_dir = get_downsampled_dir(get_data_dir(img_file))
    return os.path.join(ds_dir, get_png_name_for_jpeg(img_file))


def get_n_pass_image_file_name(img_file, out_dir, n):
    img_file = os.path.split(img_file)[1]
    img_file = os.path.splitext(img_file)[0]
    pass_n_name = os.path.join(out_dir, "%s_pass%d.png" % (img_file, n))
    return pass_n_name


def get_png_name_for_jpeg(img_file):
    img_file = os.path.split(img_file)[1]
    img_file = os.path.splitext(img_file)[0]
    return img_file + '.png'


def get_mask_file_name(img_file):
    data_dir = get_data_dir(img_file)
    masks_dir = get_masks_dir(data_dir)
    png_name = get_png_name_for_jpeg(img_file)
    return os.path.join(masks_dir, png_name)


def get_seg_file_name(img_file):
    data_dir = get_data_dir(img_file)
    seg_dir = get_segmentation_dir(data_dir)
    png_name = get_png_name_for_jpeg(img_file)
    return os.path.join(seg_dir, png_name)


def get_parts_dir_name(img_file):
    data_dir = get_data_dir(img_file)
    parts_dir = get_parts_dir(data_dir)
    dir_name = os.path.splitext(os.path.split(img_file)[1])[0]
    return os.path.join(parts_dir, dir_name)


def create_dir(d):
    if not os.path.exists(d):
        print("creating dir: %s" % d)
        os.mkdir(d)
        if not os.path.exists(d):
            raise Exception("Failed to create dir: %s" % d)


def get_data_dir(img_file):
    return os.path.split(os.path.split(img_file)[0])[0]


def get_raw_dir(data_dir):
    return os.path.join(data_dir, "raw")


def get_downsampled_dir(data_dir):
    return os.path.join(data_dir, "downsampled")


def get_masks_dir(data_dir):
    return os.path.join(data_dir, "masks")


def get_segmentation_dir(data_dir):
    return os.path.join(data_dir, "segmentation")


def get_parts_dir(data_dir):
    return os.path.join(data_dir, "parts")
