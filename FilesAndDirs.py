import os


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


def get_mask_file_name_for_image(img_file):
    img_dir, img_file = os.path.split(img_file)
    mask_dir = os.path.join(img_dir, "masks")
    mask_file_name = os.path.splitext(img_file)[0] + '.png'
    mask_file_path = os.path.join(mask_dir, mask_file_name)
    return mask_file_path


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


def get_raw_dir(data_dir):
    return os.path.join(data_dir, "raw")


def get_downsampled_dir(data_dir):
    return os.path.join(data_dir, "downsampled")


def get_masks_dir(data_dir):
    return os.path.join(data_dir, "masks")


def get_data_dir(img_file):
    return os.path.split(os.path.split(img_file)[0])[0]
