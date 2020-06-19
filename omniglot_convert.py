import os
import sys
import pickle
import glob
import numpy as np
import cv2
from common import num_from_byte_array, mkdir, f2s, FOCAL_S, FOCAL
import struct
from focal import focal_to_spike
import zipfile
import copy

def safe_dir_name(dir_in, replace_in=('(', ')', ' ',), replace_out=('-', '-', '_')):
    dir_out = dir_in
    for i, ch in enumerate(replace_in):
        dir_out = dir_out.replace(ch, replace_out[i])
    return dir_out

def extract_to_dict(input_path, file_depth=4, threshold=0.5):
    BASE, ALPHA, CHAR = range(3)

    zip_file = zipfile.ZipFile(input_path)
    d = {}
    for name in zip_file.namelist():
        split_path = name.split('/')

        if len(split_path) != file_depth:
            continue

        if split_path[-1] == '':
            continue

        if split_path[ALPHA] not in d:
            d[split_path[ALPHA]] = {}

        if split_path[CHAR] not in d[split_path[ALPHA]]:
            d[split_path[ALPHA]][split_path[CHAR]] = []

        f = zip_file.read(name)

        img = cv2.imdecode(
                np.frombuffer(f, np.uint8), 
                cv2.IMREAD_GRAYSCALE).astype('float')

        # invert image --- so high values mean 255 and low mean 0
        hi = np.where(img > threshold)
        lo = np.where(img <= threshold)
        img[lo] = 255.0
        img[hi] = 0.0
        
        d[split_path[ALPHA]][split_path[CHAR]].append(img)

    return d


def omniglot_convert(file_dict, out_dir, percent, timestep, spikes_per_bin=1, 
                     skip_existing=True, scaling=1.0):
    n_total = 0
    for alpha in file_dict:
        for char in file_dict[alpha]:
            n_total += len(file_dict[alpha][char])
    
    h, w = file_dict[alpha][char][0].shape
    if scaling != 1.0:
        sys.stdout.write('\tscaling input image from shape {}'.format((h, w)))
        h, w = int(h * scaling), int(w * scaling)
        sys.stdout.write(' to {}\n\n'.format((h, w)))
        sys.stdout.flush()

    s_img = np.zeros((h, w))
    n_processed = 0
    spikes = []
    ssa = []
    for a_idx, alpha in enumerate(sorted(file_dict.keys())):
        safe_alpha = safe_dir_name(alpha)
        _dir = os.path.join(out_dir, safe_alpha)
        mkdir(_dir)

        for ch_idx, char in enumerate(sorted(file_dict[alpha].keys())):
            _dir = os.path.join(out_dir, safe_alpha, char)
            mkdir(_dir)

            for i_idx, img in enumerate(file_dict[alpha][char]):
                pc = 100.0*float(n_processed + 1)/n_total
                sys.stdout.write('\rconverting {:6.2f}%'.format(pc))
                sys.stdout.flush()
                fname = 'alpha_{}_class_{:06d}_timestep_{}_index_{:06d}.npz'.\
                            format(safe_alpha, ch_idx, f2s(timestep), i_idx)
                out_path = os.path.join(_dir, fname)
                if os.path.isfile(out_path) and skip_existing:
                    n_processed += 1
                    continue


                if scaling != 1.0:
                    s_img[:] = np.clip(
                                cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC),
                                0.0, 255.0)
                else:
                    s_img[:] = img
                
                whr = np.where(s_img > 0)
                r0 = -(np.mean(whr[0]) - (h // 2))
                c0 = -(np.mean(whr[1]) - (w // 2))
                T = np.float32([[1, 0, c0], [0, 1, r0]])
                s_img[:] = cv2.warpAffine(s_img, T, (w, h)) 
                
                fcl = FOCAL_S if w <= 64 else FOCAL
                spikes[:] = fcl.apply(s_img, percent)
                ssa[:] = focal_to_spike(spikes, s_img.shape,          
                            spikes_per_time_block=spikes_per_bin, 
                            start_time=0., time_step=timestep)
                kernels = {k: fcl.kernels.full_kernels[k] 
                           for k in fcl.kernels.full_kernels}
                # print(kernels)
                keepers = {k: fcl.convolver.get_subsample_keepers(k) 
                           for k in kernels}
                np.savez_compressed(out_path,
                    label=ch_idx, color_image=img, grayscale_image=img, 
                    scaled_image=s_img, focal_spikes=spikes, spike_source_array=ssa, 
                    timestep=timestep, image_batch_index=i_idx, scaling=scaling,
                    kernels=kernels, alphabet=alpha, alpha_idx=a_idx, keepers=keepers)

                n_processed += 1
    sys.stdout.write("\n\ndone!\n")
    sys.stdout.flush()

def open_and_convert(in_dir, out_dir, percent, timestep, spikes_per_bin=1, 
                    skip_existing=True, scaling=1.0):
    fpath = os.path.join(in_dir, 'images_background.zip')
    files = extract_to_dict(fpath, file_depth=4, threshold=0.5)
    omniglot_convert(files, out_dir, percent, timestep, spikes_per_bin, 
                     skip_existing, scaling)

