# Serving parity summary

train env: {"python": "3.10.19", "machine": "Darwin arm64", "numpy": "2.2.6", "cv2": "4.12.0", "PIL": "11.3.0", "torch": "2.7.1", "torchvision": "0.22.1", "transformers": "4.55.0", "sidewalk_panorama_tools": "1.0.0", "libwebp": "1.5.0"}

api env: {"python": "3.12.14", "machine": "Darwin arm64", "numpy": "2.5.4", "cv2": "5.0.0", "PIL": "12.3.0", "torch": "2.14.1", "torchvision": "0.29.1", "transformers": "5.20.0", "sidewalk_panorama_tools": "1.0.0", "libwebp": "1.6.0"}

40 cases (10 points x 4 poses). `roll1_only` (a roll with no pitch) is reported apart: the API ignores a roll without a pitch.

| cases | stage | result |
|---|---|---|
| all | 1 pano read+resize, train vs API | identical=False, max 1 |
| all | 1 pano, train vs train_other1 | identical=False, max 1, frac 1.01e-01 |
| matched poses | 2 view train torch cpu vs API cpu | identical 0/30, max 2 |
| matched poses | 2 tilt_applied agrees | 30/30 |
| matched poses | 3 depth map identical train vs API | 0/30, max 2.2461 m |
| matched poses | 4 crop_half train vs API | equal 25/30 |
| matched poses | 4 crop bytes train vs API | identical 0/30, max 241, mean abs max 3.95 |
| roll1_only | 2 view train torch cpu vs API cpu | identical 0/10, max 253 |
| roll1_only | 2 tilt_applied agrees | 0/10 |
| roll1_only | 3 depth map identical train vs API | 0/10, max 52.6915 m |
| roll1_only | 4 crop_half train vs API | equal 0/10 |
| roll1_only | 4 crop bytes train vs API | identical 0/10, max 254, mean abs max 22.1 |
| matched poses | 2-4 train_other1 env vs API: view / depth map / crop identical | 30 / 30 / 30 of 30 |
| roll1_only | 2-4 train_other1 env vs API: view / depth map / crop identical | 0 / 0 / 0 of 10 |
| all | 2 view numpy vs torch cpu (train) | max 1, frac px median 1.70e-03 max 3.02e-03, centre max 1 |
| all | 2 view torch cpu vs mps (train_other1 env) | max 1, frac px max 1.58e-03, centre max 1 |
| all | 3 depth input is the BGR view (train) | 40/40 |
| all | 3 probe: channels swapped | centre depth change median 0.263 m, max 3.17 m; crop_half changed 35/40 |
| all | 3 probe: depth on mps vs cpu | centre max 3.43e-05 m; crop_half changed 0/40 (train_other1 env) |
| all | 4 API crop-half rule on the training depth | equal 40/40 |
| matched poses | probe: a channel-order bug would show as | view mean abs >= 3.68, crop mean abs >= 3.57 levels |
| all | 5 WebP q95 vs raw (pixels) | mean abs median 1.16, max abs max 25; gen2 vs gen1 mean abs median 0.224 |


## Model outputs, five models pooled (stages 2, 4, 5-7)

| comparison | cases | n | max_dlogit | median_dprob | p90_dprob | max_dprob | verdict_flips | vote_changes | votes_ref |
|---|---|---|---|---|---|---|---|---|---|
| 2_mps_vs_cpu_render (raw crop, train_other1 env) | matched poses | 150 | 0.1115 | 1.397e-05 | 0.0004447 | 0.001561 | 0 | 0 | 19 |
| 2_mps_vs_cpu_render (raw crop, train_other1 env) | roll1_only | 50 | 0.005664 | 6.02e-06 | 0.0002674 | 0.002012 | 0 | 0 | 7 |
| 2_numpy_vs_torch_render (raw crop, train env) | matched poses | 150 | 0.09866 | 1.565e-05 | 0.000565 | 0.007286 | 1 | 0 | 19 |
| 2_numpy_vs_torch_render (raw crop, train env) | roll1_only | 50 | 0.07287 | 1.206e-05 | 0.0004011 | 0.01471 | 0 | 0 | 7 |
| 2_numpy_vs_torch_render (raw crop, train_other1 env) | matched poses | 150 | 0.1324 | 2.016e-05 | 0.0006041 | 0.00375 | 0 | 0 | 19 |
| 2_numpy_vs_torch_render (raw crop, train_other1 env) | roll1_only | 50 | 0.01954 | 1.27e-05 | 0.0005194 | 0.002611 | 0 | 0 | 7 |
| 4_crop_half_plus1 (raw crop, train env) | matched poses | 150 | 0.6297 | 0.001302 | 0.04923 | 0.1901 | 0 | 0 | 19 |
| 4_crop_half_plus1 (raw crop, train env) | roll1_only | 50 | 0.4447 | 0.0007285 | 0.02171 | 0.07753 | 1 | 0 | 7 |
| 4_crop_half_plus1 (raw crop, train_other1 env) | matched poses | 150 | 0.5457 | 0.0008913 | 0.03616 | 0.1793 | 0 | 0 | 19 |
| 4_crop_half_plus1 (raw crop, train_other1 env) | roll1_only | 50 | 0.2769 | 0.0007854 | 0.01666 | 0.0508 | 2 | 0 | 7 |
| 5_webp2_vs_webp (train env) | matched poses | 150 | 0.2591 | 0.0003186 | 0.00975 | 0.1173 | 1 | 0 | 21 |
| 5_webp2_vs_webp (train env) | roll1_only | 50 | 0.2609 | 0.0007145 | 0.008096 | 0.06319 | 0 | 0 | 7 |
| 5_webp_vs_raw (train env) | matched poses | 150 | 0.4271 | 0.0004259 | 0.02547 | 0.1087 | 1 | 2 | 19 |
| 5_webp_vs_raw (train env) | roll1_only | 50 | 0.3779 | 0.0003061 | 0.0156 | 0.05109 | 2 | 0 | 7 |
| 6+7_model_env (same raw crop: train env vs API env) | matched poses | 150 | 0.0002885 | 2.177e-08 | 9.73e-07 | 1.673e-05 | 0 | 0 | 19 |
| 6+7_model_env (same raw crop: train env vs API env) | roll1_only | 50 | 4.154e-05 | 1.688e-08 | 7.749e-07 | 7.853e-06 | 0 | 0 | 7 |
| 7_end_to_end (train webp in train env vs API) | matched poses | 150 | 0.3827 | 0.0007633 | 0.02593 | 0.08145 | 1 | 2 | 21 |
| 7_end_to_end (train webp in train env vs API) | roll1_only | 50 | 2.127 | 0.001704 | 0.1578 | 0.3365 | 0 | 0 | 7 |
| 7_raw_crops (train raw vs API raw, API env) | matched poses | 150 | 0.2041 | 0.0003397 | 0.01092 | 0.06425 | 0 | 0 | 19 |
| 7_raw_crops (train raw vs API raw, API env) | roll1_only | 50 | 2.167 | 0.001977 | 0.1543 | 0.352 | 2 | 0 | 7 |
| 7_raw_end_to_end (train_other1 env vs API) | matched poses | 150 | 0 | 0 | 0 | 0 | 0 | 0 | 19 |
| 7_raw_end_to_end (train_other1 env vs API) | roll1_only | 50 | 2.141 | 0.002092 | 0.1388 | 0.3427 | 1 | 0 | 7 |
| env_train_other1 (webp crop, own env end to end) | matched poses | 150 | 0.4433 | 0.0005792 | 0.01872 | 0.0618 | 0 | 0 | 21 |
| env_train_other1 (webp crop, own env end to end) | roll1_only | 50 | 0.2767 | 0.0005507 | 0.0293 | 0.1047 | 0 | 0 | 7 |


## Model outputs per model (stages 2, 4, 5-7)

| comparison | cases | model | n | max_dlogit | median_dprob | p90_dprob | max_dprob | verdict_flips | vote_changes | votes_ref |
|---|---|---|---|---|---|---|---|---|---|---|
| 2_mps_vs_cpu_render (raw crop, train_other1 env) | matched poses | crosswalk | 30 | 0.006191 | 6.389e-07 | 0.000188 | 0.001561 | 0 | 0 | 10 |
| 2_mps_vs_cpu_render (raw crop, train_other1 env) | matched poses | curbramp | 30 | 0.1115 | 1.805e-05 | 0.0006713 | 0.001398 | 0 | 0 | 9 |
| 2_mps_vs_cpu_render (raw crop, train_other1 env) | matched poses | nocurbramp | 30 | 0.002272 | 0.0001581 | 0.0005222 | 0.0006466 | 0 | 0 | 0 |
| 2_mps_vs_cpu_render (raw crop, train_other1 env) | matched poses | obstacle | 30 | 0.004729 | 1.009e-05 | 0.0004625 | 0.0008816 | 0 | 0 | 0 |
| 2_mps_vs_cpu_render (raw crop, train_other1 env) | matched poses | surfaceproblem | 30 | 0.005715 | 1.827e-06 | 8.25e-05 | 0.0005048 | 0 | 0 | 0 |
| 2_mps_vs_cpu_render (raw crop, train_other1 env) | roll1_only | crosswalk | 10 | 0.001987 | 1.494e-07 | 0.0003555 | 0.0004397 | 0 | 0 | 4 |
| 2_mps_vs_cpu_render (raw crop, train_other1 env) | roll1_only | curbramp | 10 | 0.004655 | 1.018e-05 | 0.0007162 | 0.002012 | 0 | 0 | 3 |
| 2_mps_vs_cpu_render (raw crop, train_other1 env) | roll1_only | nocurbramp | 10 | 0.002377 | 8.205e-05 | 0.0002361 | 0.0005591 | 0 | 0 | 0 |
| 2_mps_vs_cpu_render (raw crop, train_other1 env) | roll1_only | obstacle | 10 | 0.002058 | 2.869e-06 | 9.384e-05 | 0.0001809 | 0 | 0 | 0 |
| 2_mps_vs_cpu_render (raw crop, train_other1 env) | roll1_only | surfaceproblem | 10 | 0.005664 | 2.007e-06 | 0.0001166 | 0.0001329 | 0 | 0 | 0 |
| 2_numpy_vs_torch_render (raw crop, train env) | matched poses | crosswalk | 30 | 0.006614 | 7.801e-07 | 0.000165 | 0.001633 | 0 | 0 | 10 |
| 2_numpy_vs_torch_render (raw crop, train env) | matched poses | curbramp | 30 | 0.05814 | 1.201e-05 | 0.0007718 | 0.00196 | 0 | 0 | 9 |
| 2_numpy_vs_torch_render (raw crop, train env) | matched poses | nocurbramp | 30 | 0.01572 | 0.0001224 | 0.0005964 | 0.001647 | 0 | 0 | 0 |
| 2_numpy_vs_torch_render (raw crop, train env) | matched poses | obstacle | 30 | 0.09866 | 2.107e-05 | 0.0007031 | 0.007286 | 0 | 0 | 0 |
| 2_numpy_vs_torch_render (raw crop, train env) | matched poses | surfaceproblem | 30 | 0.009128 | 6.263e-06 | 0.0002328 | 0.00059 | 1 | 0 | 0 |
| 2_numpy_vs_torch_render (raw crop, train env) | roll1_only | crosswalk | 10 | 0.07287 | 1.628e-06 | 0.001828 | 0.01471 | 0 | 0 | 4 |
| 2_numpy_vs_torch_render (raw crop, train env) | roll1_only | curbramp | 10 | 0.004021 | 8.516e-06 | 0.0008477 | 0.001699 | 0 | 0 | 3 |
| 2_numpy_vs_torch_render (raw crop, train env) | roll1_only | nocurbramp | 10 | 0.003417 | 0.0001866 | 0.0003127 | 0.0003704 | 0 | 0 | 0 |
| 2_numpy_vs_torch_render (raw crop, train env) | roll1_only | obstacle | 10 | 0.004283 | 2.47e-05 | 0.0002699 | 0.0007769 | 0 | 0 | 0 |
| 2_numpy_vs_torch_render (raw crop, train env) | roll1_only | surfaceproblem | 10 | 0.005021 | 9.777e-06 | 7.053e-05 | 0.0002122 | 0 | 0 | 0 |
| 2_numpy_vs_torch_render (raw crop, train_other1 env) | matched poses | crosswalk | 30 | 0.08322 | 1.431e-06 | 0.0002164 | 0.0006538 | 0 | 0 | 10 |
| 2_numpy_vs_torch_render (raw crop, train_other1 env) | matched poses | curbramp | 30 | 0.1324 | 2.217e-05 | 0.001027 | 0.00375 | 0 | 0 | 9 |
| 2_numpy_vs_torch_render (raw crop, train_other1 env) | matched poses | nocurbramp | 30 | 0.007173 | 0.0002156 | 0.0006578 | 0.00136 | 0 | 0 | 0 |
| 2_numpy_vs_torch_render (raw crop, train_other1 env) | matched poses | obstacle | 30 | 0.006104 | 1.348e-05 | 0.0004039 | 0.001532 | 0 | 0 | 0 |
| 2_numpy_vs_torch_render (raw crop, train_other1 env) | matched poses | surfaceproblem | 30 | 0.09937 | 9.447e-06 | 0.0002711 | 0.001808 | 0 | 0 | 0 |
| 2_numpy_vs_torch_render (raw crop, train_other1 env) | roll1_only | crosswalk | 10 | 0.002915 | 8.417e-07 | 4.486e-05 | 0.0003515 | 0 | 0 | 4 |
| 2_numpy_vs_torch_render (raw crop, train_other1 env) | roll1_only | curbramp | 10 | 0.01663 | 3.444e-05 | 0.001574 | 0.002611 | 0 | 0 | 3 |
| 2_numpy_vs_torch_render (raw crop, train_other1 env) | roll1_only | nocurbramp | 10 | 0.005308 | 8.374e-05 | 0.0007006 | 0.002451 | 0 | 0 | 0 |
| 2_numpy_vs_torch_render (raw crop, train_other1 env) | roll1_only | obstacle | 10 | 0.006472 | 3.189e-05 | 0.0006776 | 0.001023 | 0 | 0 | 0 |
| 2_numpy_vs_torch_render (raw crop, train_other1 env) | roll1_only | surfaceproblem | 10 | 0.01954 | 2.307e-05 | 0.0001325 | 0.0003201 | 0 | 0 | 0 |
| 4_crop_half_plus1 (raw crop, train env) | matched poses | crosswalk | 30 | 0.252 | 3.72e-05 | 0.004935 | 0.06346 | 0 | 0 | 10 |
| 4_crop_half_plus1 (raw crop, train env) | matched poses | curbramp | 30 | 0.5102 | 0.001614 | 0.05151 | 0.1115 | 0 | 0 | 9 |
| 4_crop_half_plus1 (raw crop, train env) | matched poses | nocurbramp | 30 | 0.3543 | 0.0129 | 0.06245 | 0.07797 | 0 | 0 | 0 |
| 4_crop_half_plus1 (raw crop, train env) | matched poses | obstacle | 30 | 0.5153 | 0.001041 | 0.05182 | 0.1247 | 0 | 0 | 0 |
| 4_crop_half_plus1 (raw crop, train env) | matched poses | surfaceproblem | 30 | 0.6297 | 0.0002436 | 0.0124 | 0.1901 | 0 | 0 | 0 |
| 4_crop_half_plus1 (raw crop, train env) | roll1_only | crosswalk | 10 | 0.2424 | 8.413e-05 | 0.02396 | 0.04755 | 0 | 0 | 4 |
| 4_crop_half_plus1 (raw crop, train env) | roll1_only | curbramp | 10 | 0.1461 | 0.0002677 | 0.02321 | 0.03816 | 0 | 0 | 3 |
| 4_crop_half_plus1 (raw crop, train env) | roll1_only | nocurbramp | 10 | 0.2728 | 0.009503 | 0.0264 | 0.05533 | 1 | 0 | 0 |
| 4_crop_half_plus1 (raw crop, train env) | roll1_only | obstacle | 10 | 0.2318 | 0.0002659 | 0.01223 | 0.07753 | 0 | 0 | 0 |
| 4_crop_half_plus1 (raw crop, train env) | roll1_only | surfaceproblem | 10 | 0.4447 | 0.0003117 | 0.01509 | 0.01566 | 0 | 0 | 0 |
| 4_crop_half_plus1 (raw crop, train_other1 env) | matched poses | crosswalk | 30 | 0.2583 | 4.454e-05 | 0.01077 | 0.0387 | 0 | 0 | 10 |
| 4_crop_half_plus1 (raw crop, train_other1 env) | matched poses | curbramp | 30 | 0.5414 | 0.001522 | 0.04317 | 0.1623 | 0 | 0 | 9 |
| 4_crop_half_plus1 (raw crop, train_other1 env) | matched poses | nocurbramp | 30 | 0.2993 | 0.01045 | 0.06156 | 0.06837 | 0 | 0 | 0 |
| 4_crop_half_plus1 (raw crop, train_other1 env) | matched poses | obstacle | 30 | 0.5266 | 0.0007164 | 0.03977 | 0.1322 | 0 | 0 | 0 |
| 4_crop_half_plus1 (raw crop, train_other1 env) | matched poses | surfaceproblem | 30 | 0.5457 | 0.0001941 | 0.01051 | 0.1793 | 0 | 0 | 0 |
| 4_crop_half_plus1 (raw crop, train_other1 env) | roll1_only | crosswalk | 10 | 0.2038 | 2.896e-05 | 0.01235 | 0.01469 | 0 | 0 | 4 |
| 4_crop_half_plus1 (raw crop, train_other1 env) | roll1_only | curbramp | 10 | 0.1481 | 0.002414 | 0.01238 | 0.01668 | 0 | 0 | 3 |
| 4_crop_half_plus1 (raw crop, train_other1 env) | roll1_only | nocurbramp | 10 | 0.2716 | 0.009715 | 0.03 | 0.05036 | 2 | 0 | 0 |
| 4_crop_half_plus1 (raw crop, train_other1 env) | roll1_only | obstacle | 10 | 0.1679 | 0.000287 | 0.009277 | 0.0508 | 0 | 0 | 0 |
| 4_crop_half_plus1 (raw crop, train_other1 env) | roll1_only | surfaceproblem | 10 | 0.2769 | 0.0001565 | 0.003925 | 0.01664 | 0 | 0 | 0 |
| 5_webp2_vs_webp (train env) | matched poses | crosswalk | 30 | 0.04853 | 4.429e-05 | 0.00385 | 0.009658 | 0 | 0 | 10 |
| 5_webp2_vs_webp (train env) | matched poses | curbramp | 30 | 0.2591 | 0.0005592 | 0.02572 | 0.1173 | 1 | 0 | 11 |
| 5_webp2_vs_webp (train env) | matched poses | nocurbramp | 30 | 0.1756 | 0.002532 | 0.01959 | 0.03686 | 0 | 0 | 0 |
| 5_webp2_vs_webp (train env) | matched poses | obstacle | 30 | 0.09048 | 0.0001849 | 0.007683 | 0.01525 | 0 | 0 | 0 |
| 5_webp2_vs_webp (train env) | matched poses | surfaceproblem | 30 | 0.1516 | 9.088e-05 | 0.005609 | 0.01129 | 0 | 0 | 0 |
| 5_webp2_vs_webp (train env) | roll1_only | crosswalk | 10 | 0.07066 | 3.574e-05 | 0.004317 | 0.005458 | 0 | 0 | 4 |
| 5_webp2_vs_webp (train env) | roll1_only | curbramp | 10 | 0.2609 | 0.0008664 | 0.03844 | 0.06319 | 0 | 0 | 3 |
| 5_webp2_vs_webp (train env) | roll1_only | nocurbramp | 10 | 0.06362 | 0.003957 | 0.008096 | 0.009595 | 0 | 0 | 0 |
| 5_webp2_vs_webp (train env) | roll1_only | obstacle | 10 | 0.09731 | 0.001237 | 0.007756 | 0.01231 | 0 | 0 | 0 |
| 5_webp2_vs_webp (train env) | roll1_only | surfaceproblem | 10 | 0.1818 | 0.0001275 | 0.005804 | 0.02371 | 0 | 0 | 0 |
| 5_webp_vs_raw (train env) | matched poses | crosswalk | 30 | 0.2408 | 2.423e-05 | 0.005526 | 0.05713 | 0 | 0 | 10 |
| 5_webp_vs_raw (train env) | matched poses | curbramp | 30 | 0.4271 | 0.001818 | 0.02562 | 0.1087 | 0 | 2 | 9 |
| 5_webp_vs_raw (train env) | matched poses | nocurbramp | 30 | 0.304 | 0.008663 | 0.03397 | 0.04336 | 0 | 0 | 0 |
| 5_webp_vs_raw (train env) | matched poses | obstacle | 30 | 0.2079 | 0.0001099 | 0.01937 | 0.03806 | 0 | 0 | 0 |
| 5_webp_vs_raw (train env) | matched poses | surfaceproblem | 30 | 0.2016 | 0.0002244 | 0.02707 | 0.05078 | 1 | 0 | 0 |
| 5_webp_vs_raw (train env) | roll1_only | crosswalk | 10 | 0.2591 | 2.909e-05 | 0.01312 | 0.05109 | 0 | 0 | 4 |
| 5_webp_vs_raw (train env) | roll1_only | curbramp | 10 | 0.1622 | 0.001142 | 0.01403 | 0.01665 | 0 | 0 | 3 |
| 5_webp_vs_raw (train env) | roll1_only | nocurbramp | 10 | 0.0935 | 0.007286 | 0.02413 | 0.03067 | 2 | 0 | 0 |
| 5_webp_vs_raw (train env) | roll1_only | obstacle | 10 | 0.1639 | 0.0001231 | 0.007027 | 0.01549 | 0 | 0 | 0 |
| 5_webp_vs_raw (train env) | roll1_only | surfaceproblem | 10 | 0.3779 | 0.0002485 | 0.009581 | 0.01442 | 0 | 0 | 0 |
| 6+7_model_env (same raw crop: train env vs API env) | matched poses | crosswalk | 30 | 1.35e-05 | 1.63e-09 | 7.331e-07 | 3.815e-06 | 0 | 0 | 10 |
| 6+7_model_env (same raw crop: train env vs API env) | matched poses | curbramp | 30 | 0.0001301 | 1.676e-08 | 1.514e-06 | 4.275e-06 | 0 | 0 | 9 |
| 6+7_model_env (same raw crop: train env vs API env) | matched poses | nocurbramp | 30 | 2.098e-05 | 3.353e-07 | 8.047e-07 | 1.609e-06 | 0 | 0 | 0 |
| 6+7_model_env (same raw crop: train env vs API env) | matched poses | obstacle | 30 | 0.0002885 | 8.626e-08 | 1.025e-06 | 1.673e-05 | 0 | 0 | 0 |
| 6+7_model_env (same raw crop: train env vs API env) | matched poses | surfaceproblem | 30 | 1.001e-05 | 6.985e-09 | 7.004e-07 | 1.132e-06 | 0 | 0 | 0 |
| 6+7_model_env (same raw crop: train env vs API env) | roll1_only | crosswalk | 10 | 4.154e-05 | 1.251e-09 | 1.161e-06 | 7.853e-06 | 0 | 0 | 4 |
| 6+7_model_env (same raw crop: train env vs API env) | roll1_only | curbramp | 10 | 4.172e-06 | 6.612e-08 | 8.434e-07 | 1.46e-06 | 0 | 0 | 3 |
| 6+7_model_env (same raw crop: train env vs API env) | roll1_only | nocurbramp | 10 | 3.919e-06 | 2.384e-07 | 6.676e-07 | 7.749e-07 | 0 | 0 | 0 |
| 6+7_model_env (same raw crop: train env vs API env) | roll1_only | obstacle | 10 | 8.106e-06 | 6.671e-08 | 8.084e-07 | 1.311e-06 | 0 | 0 | 0 |
| 6+7_model_env (same raw crop: train env vs API env) | roll1_only | surfaceproblem | 10 | 5.841e-06 | 1.251e-09 | 3.55e-07 | 1.371e-06 | 0 | 0 | 0 |
| 7_end_to_end (train webp in train env vs API) | matched poses | crosswalk | 30 | 0.3827 | 5.598e-05 | 0.005425 | 0.05281 | 0 | 0 | 10 |
| 7_end_to_end (train webp in train env vs API) | matched poses | curbramp | 30 | 0.3418 | 0.0012 | 0.03738 | 0.08145 | 0 | 2 | 11 |
| 7_end_to_end (train webp in train env vs API) | matched poses | nocurbramp | 30 | 0.3255 | 0.006454 | 0.02993 | 0.05339 | 0 | 0 | 0 |
| 7_end_to_end (train webp in train env vs API) | matched poses | obstacle | 30 | 0.346 | 0.0003256 | 0.01633 | 0.05662 | 0 | 0 | 0 |
| 7_end_to_end (train webp in train env vs API) | matched poses | surfaceproblem | 30 | 0.2523 | 0.0002695 | 0.02767 | 0.06538 | 1 | 0 | 0 |
| 7_end_to_end (train webp in train env vs API) | roll1_only | crosswalk | 10 | 0.3327 | 0.0002848 | 0.02814 | 0.04108 | 0 | 0 | 4 |
| 7_end_to_end (train webp in train env vs API) | roll1_only | curbramp | 10 | 1.287 | 0.001889 | 0.1578 | 0.1627 | 0 | 0 | 3 |
| 7_end_to_end (train webp in train env vs API) | roll1_only | nocurbramp | 10 | 1.087 | 0.03672 | 0.2106 | 0.2954 | 0 | 0 | 0 |
| 7_end_to_end (train webp in train env vs API) | roll1_only | obstacle | 10 | 2.127 | 0.003767 | 0.2306 | 0.3365 | 0 | 0 | 0 |
| 7_end_to_end (train webp in train env vs API) | roll1_only | surfaceproblem | 10 | 0.9241 | 0.001194 | 0.07023 | 0.1274 | 0 | 0 | 0 |
| 7_raw_crops (train raw vs API raw, API env) | matched poses | crosswalk | 30 | 0.1419 | 3.329e-05 | 0.002627 | 0.01159 | 0 | 0 | 10 |
| 7_raw_crops (train raw vs API raw, API env) | matched poses | curbramp | 30 | 0.2041 | 0.0008594 | 0.01738 | 0.03808 | 0 | 0 | 9 |
| 7_raw_crops (train raw vs API raw, API env) | matched poses | nocurbramp | 30 | 0.1705 | 0.002501 | 0.007237 | 0.06425 | 0 | 0 | 0 |
| 7_raw_crops (train raw vs API raw, API env) | matched poses | obstacle | 30 | 0.1705 | 0.0002538 | 0.004891 | 0.01856 | 0 | 0 | 0 |
| 7_raw_crops (train raw vs API raw, API env) | matched poses | surfaceproblem | 30 | 0.2008 | 5.657e-05 | 0.01094 | 0.03844 | 0 | 0 | 0 |
| 7_raw_crops (train raw vs API raw, API env) | roll1_only | crosswalk | 10 | 0.431 | 0.0003197 | 0.02524 | 0.09216 | 0 | 0 | 4 |
| 7_raw_crops (train raw vs API raw, API env) | roll1_only | curbramp | 10 | 1.188 | 0.00301 | 0.1543 | 0.1793 | 0 | 0 | 3 |
| 7_raw_crops (train raw vs API raw, API env) | roll1_only | nocurbramp | 10 | 1.133 | 0.03953 | 0.2144 | 0.3188 | 2 | 0 | 0 |
| 7_raw_crops (train raw vs API raw, API env) | roll1_only | obstacle | 10 | 2.167 | 0.0007703 | 0.2376 | 0.352 | 0 | 0 | 0 |
| 7_raw_crops (train raw vs API raw, API env) | roll1_only | surfaceproblem | 10 | 0.9245 | 0.001487 | 0.05816 | 0.1365 | 0 | 0 | 0 |
| 7_raw_end_to_end (train_other1 env vs API) | matched poses | crosswalk | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 10 |
| 7_raw_end_to_end (train_other1 env vs API) | matched poses | curbramp | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 9 |
| 7_raw_end_to_end (train_other1 env vs API) | matched poses | nocurbramp | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7_raw_end_to_end (train_other1 env vs API) | matched poses | obstacle | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7_raw_end_to_end (train_other1 env vs API) | matched poses | surfaceproblem | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 7_raw_end_to_end (train_other1 env vs API) | roll1_only | crosswalk | 10 | 0.4348 | 0.0002656 | 0.01505 | 0.05617 | 0 | 0 | 4 |
| 7_raw_end_to_end (train_other1 env vs API) | roll1_only | curbramp | 10 | 1.146 | 0.003411 | 0.1388 | 0.1877 | 0 | 0 | 3 |
| 7_raw_end_to_end (train_other1 env vs API) | roll1_only | nocurbramp | 10 | 1.123 | 0.02835 | 0.2123 | 0.3053 | 1 | 0 | 0 |
| 7_raw_end_to_end (train_other1 env vs API) | roll1_only | obstacle | 10 | 2.141 | 0.001062 | 0.237 | 0.3427 | 0 | 0 | 0 |
| 7_raw_end_to_end (train_other1 env vs API) | roll1_only | surfaceproblem | 10 | 0.8623 | 0.001596 | 0.05641 | 0.1271 | 0 | 0 | 0 |
| env_train_other1 (webp crop, own env end to end) | matched poses | crosswalk | 30 | 0.154 | 5.05e-05 | 0.004848 | 0.03628 | 0 | 0 | 10 |
| env_train_other1 (webp crop, own env end to end) | matched poses | curbramp | 30 | 0.2528 | 0.001186 | 0.01767 | 0.0618 | 0 | 0 | 11 |
| env_train_other1 (webp crop, own env end to end) | matched poses | nocurbramp | 30 | 0.2258 | 0.007411 | 0.02492 | 0.05699 | 0 | 0 | 0 |
| env_train_other1 (webp crop, own env end to end) | matched poses | obstacle | 30 | 0.2099 | 0.0004088 | 0.02191 | 0.04014 | 0 | 0 | 0 |
| env_train_other1 (webp crop, own env end to end) | matched poses | surfaceproblem | 30 | 0.4433 | 0.0002852 | 0.01203 | 0.03515 | 0 | 0 | 0 |
| env_train_other1 (webp crop, own env end to end) | roll1_only | crosswalk | 10 | 0.12 | 6.411e-05 | 0.004374 | 0.01224 | 0 | 0 | 4 |
| env_train_other1 (webp crop, own env end to end) | roll1_only | curbramp | 10 | 0.2767 | 0.001513 | 0.04903 | 0.1047 | 0 | 0 | 3 |
| env_train_other1 (webp crop, own env end to end) | roll1_only | nocurbramp | 10 | 0.1368 | 0.005558 | 0.01476 | 0.02703 | 0 | 0 | 0 |
| env_train_other1 (webp crop, own env end to end) | roll1_only | obstacle | 10 | 0.1586 | 0.0005326 | 0.01441 | 0.03959 | 0 | 0 | 0 |
| env_train_other1 (webp crop, own env end to end) | roll1_only | surfaceproblem | 10 | 0.2708 | 8.133e-05 | 0.004312 | 0.02816 | 0 | 0 | 0 |


## Preprocessing (stage 6)

| comparison | model | n | max_dpixel |
|---|---|---|---|
| 6_control (4.55 fast vs 4.55 slow) | crosswalk | 40 | 0.01751 |
| 6_control (4.55 fast vs 4.55 slow) | curbramp | 40 | 0.01751 |
| 6_control (4.55 fast vs 4.55 slow) | nocurbramp | 40 | 0.01751 |
| 6_control (4.55 fast vs 4.55 slow) | obstacle | 40 | 0.01751 |
| 6_control (4.55 fast vs 4.55 slow) | surfaceproblem | 40 | 0.01751 |
| 6_control (API default backend vs pil) | crosswalk | 40 | 0.01751 |
| 6_control (API default backend vs pil) | curbramp | 40 | 0.01751 |
| 6_control (API default backend vs pil) | nocurbramp | 40 | 0.01751 |
| 6_control (API default backend vs pil) | obstacle | 40 | 0.01751 |
| 6_control (API default backend vs pil) | surfaceproblem | 40 | 0.01751 |
| 6_env_train_other1 pixel_values (own raw crop) | crosswalk | 40 | 2.504 |
| 6_env_train_other1 pixel_values (own raw crop) | curbramp | 40 | 2.504 |
| 6_env_train_other1 pixel_values (own raw crop) | nocurbramp | 40 | 2.504 |
| 6_env_train_other1 pixel_values (own raw crop) | obstacle | 40 | 2.504 |
| 6_env_train_other1 pixel_values (own raw crop) | surfaceproblem | 40 | 2.504 |
| 6_pixel_values (4.55 slow vs API pil) | crosswalk | 40 | 0 |
| 6_pixel_values (4.55 slow vs API pil) | curbramp | 40 | 0 |
| 6_pixel_values (4.55 slow vs API pil) | nocurbramp | 40 | 0 |
| 6_pixel_values (4.55 slow vs API pil) | obstacle | 40 | 0 |
| 6_pixel_values (4.55 slow vs API pil) | surfaceproblem | 40 | 0 |


API `validate()` verdict/confidence reproduced by the recorder: 200/200

## isolation.txt

```
train env: {"python": "3.10.19", "machine": "Darwin arm64", "numpy": "2.2.6", "cv2": "4.12.0", "PIL": "11.3.0", "torch": "2.7.1", "torchvision": "0.22.1", "transformers": "4.55.0", "sidewalk_panorama_tools": "1.0.0", "libwebp": "1.5.0"}
api env: {"python": "3.12.14", "machine": "Darwin arm64", "numpy": "2.5.4", "cv2": "5.0.0", "PIL": "12.3.0", "torch": "2.14.1", "torchvision": "0.29.1", "transformers": "5.20.0", "sidewalk_panorama_tools": "1.0.0", "libwebp": "1.6.0"}
api fed: pano_bgr.npy (the train worker's read+resize)
matched poses: view identical 30/30 (max 0, px frac max 0.00e+00); depth identical 0/30 (max 0.00183 m); crop_half equal 30/30; crop identical 30/30 (mean abs max 0)
roll1_only: view identical 0/10 (max 253, px frac max 9.94e-01); depth identical 0/10 (max 52.7 m); crop_half equal 0/10; crop identical 0/10 (mean abs max 22.1)
```

## pixel_probe.txt

```
a: {"python": "3.10.19", "machine": "Darwin arm64", "numpy": "2.2.6", "cv2": "4.12.0", "PIL": "11.3.0", "torch": "2.7.1", "torchvision": "0.22.1", "transformers": "4.55.0", "sidewalk_panorama_tools": "1.0.0", "libwebp": "1.5.0"}
b: {"python": "3.12.14", "machine": "Darwin arm64", "numpy": "2.5.4", "cv2": "5.0.0", "PIL": "12.3.0", "torch": "2.14.1", "torchvision": "0.29.1", "transformers": "5.20.0", "sidewalk_panorama_tools": "1.0.0", "libwebp": "1.6.0"}
decode: identical=True max_abs=0 frac_values=0.0000
resize_linear: identical=False max_abs=1 frac_values=0.1013
resize_cubic: identical=True max_abs=0 frac_values=0.0000
webp q95 bytes identical: 40/40 crops (PIL 11.3.0 libwebp 1.5.0 vs PIL 12.3.0 libwebp 1.6.0)
```
