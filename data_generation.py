import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--label_type', required=True, help='Specify the label type')
args = parser.parse_args()

label_type = args.label_type

import pandas as pd
import io
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import requests
import torch
import torch.nn.functional as F
import os
import uuid
import sys
import time
from datetime import timedelta

sys.path.insert(0, 'Depth-Anything-V2/metric_depth')
from depth_anything_v2.dpt import DepthAnythingV2

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}

encoder = 'vitl'
dataset = 'vkitti'
max_depth = 80

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load(f'depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cuda'))
model.to('cuda').eval()

cdmx = pd.read_csv('https://sidewalk-cdmx.cs.washington.edu/v3/api/rawLabels?filetype=csv')
cdmx['city'] = "cdmx"
la_piedad = pd.read_csv('https://sidewalk-la-piedad.cs.washington.edu/v3/api/rawLabels?filetype=csv')
la_piedad['city'] = "la_piedad"
spgg = pd.read_csv('https://sidewalk-spgg.cs.washington.edu/v3/api/rawLabels?filetype=csv')
spgg['city'] = "spgg"
amsterdam = pd.read_csv('https://sidewalk-amsterdam.cs.washington.edu/v3/api/rawLabels?filetype=csv')
amsterdam['city'] = "amsterdam"
blackhawk_hills = pd.read_csv('https://sidewalk-blackhawk-hills.cs.washington.edu/v3/api/rawLabels?filetype=csv')
blackhawk_hills['city'] = "blackhawk_hills"
# Hot fix. See https://github.com/ProjectSidewalk/SidewalkWebpage/issues/3756
chicago = pd.read_csv(io.StringIO(requests.get('https://sidewalk-chicago.cs.washington.edu/v3/api/rawLabels?filetype=csv').text.replace("Little Italy, UIC", "").replace("Sauganash,Forest Glen", "")))
chicago['city'] = "chicago"
cliffside_park = pd.read_csv('https://sidewalk-cliffside-park.cs.washington.edu/v3/api/rawLabels?filetype=csv')
cliffside_park['city'] = "cliffside_park"
columbus = pd.read_csv('https://sidewalk-columbus.cs.washington.edu/v3/api/rawLabels?filetype=csv')
columbus['city'] = "columbus"
knox = pd.read_csv('https://sidewalk-knox.cs.washington.edu/v3/api/rawLabels?filetype=csv')
knox['city'] = "knox"
mendota = pd.read_csv('https://sidewalk-mendota.cs.washington.edu/v3/api/rawLabels?filetype=csv')
mendota['city'] = "mendota"
newberg = pd.read_csv('https://sidewalk-newberg.cs.washington.edu/v3/api/rawLabels?filetype=csv')
newberg['city'] = "newberg"
oradell = pd.read_csv('https://sidewalk-oradell.cs.washington.edu/v3/api/rawLabels?filetype=csv')
oradell['city'] = "oradell"
pittsburgh = pd.read_csv('https://sidewalk-pittsburgh.cs.washington.edu/v3/api/rawLabels?filetype=csv')
pittsburgh['city'] = "pittsburgh"
sea = pd.read_csv('https://sidewalk-sea.cs.washington.edu/v3/api/rawLabels?filetype=csv')
sea['city'] = "sea"
st_louis = pd.read_csv('https://sidewalk-st-louis.cs.washington.edu/v3/api/rawLabels?filetype=csv')
st_louis['city'] = "st_louis"
teaneck = pd.read_csv('https://sidewalk-teaneck.cs.washington.edu/v3/api/rawLabels?filetype=csv')
teaneck['city'] = "teaneck"
keelung = pd.read_csv('https://sidewalk-keelung.cs.washington.edu/v3/api/rawLabels?filetype=csv')
keelung['city'] = "keelung"
new_taipei = pd.read_csv('https://sidewalk-new-taipei.cs.washington.edu/v3/api/rawLabels?filetype=csv')
new_taipei['city'] = "new_taipei"
taipei = pd.read_csv('https://sidewalk-taipei.cs.washington.edu/v3/api/rawLabels?filetype=csv')
taipei['city'] = "taipei"

all_cities = pd.concat([cdmx, la_piedad, spgg, amsterdam, blackhawk_hills, chicago,
                        cliffside_park, columbus, knox, mendota, newberg, oradell,
                        pittsburgh, sea, st_louis, teaneck, keelung, new_taipei, taipei])

all_cities = all_cities[all_cities['label_type'].isin([label_type])]

CORRECT_PROPORTION = 0.55

filtered_disagree = all_cities[
    all_cities['disagree_count'] - all_cities['agree_count'] > 1
].copy()
filtered_disagree['label_class'] = 'incorrect'
print(f"Incorrect Examples: {len(filtered_disagree)}")

filtered_agree = all_cities[
    all_cities['agree_count'] - all_cities['disagree_count'] > 2
].copy()
filtered_agree['label_class'] = 'correct'
filtered_agree = filtered_agree.sample(frac=1).reset_index(drop=True)
amount_correct = int((CORRECT_PROPORTION / (1 - CORRECT_PROPORTION)) * len(filtered_disagree))
filtered_agree = filtered_agree[:amount_correct]
print(f"Correct Examples: {len(filtered_agree)}")

combined_filtered = pd.concat([filtered_agree, filtered_disagree])
combined_filtered = combined_filtered.sample(frac=1).reset_index(drop=True) # Shuffle the combined dataframe

def fetch_panorama(pano_id):
    chosen_zoom = None

    def _fetch_tile(x, y, zoom=4):
        nonlocal chosen_zoom

        if chosen_zoom != None:
            zoom = chosen_zoom
        
        url = (
            f"https://streetviewpixels-pa.googleapis.com/v1/tile"
            f"?cb_client=maps_sv.tactile&panoid={pano_id}"
            f"&x={x}&y={y}&zoom={zoom}"
        )
        try:
            response = requests.get(url)
            if response.status_code == 200:
                tile = Image.open(io.BytesIO(response.content))
                if chosen_zoom != None or not _is_black_tile(tile):
                    chosen_zoom = zoom
                    return x, y, tile
        except Exception as e:
            print(e)

        if chosen_zoom == None:
            # Try fallback with zoom=3
            fallback_url = (
                f"https://streetviewpixels-pa.googleapis.com/v1/tile"
                f"?cb_client=maps_sv.tactile&panoid={pano_id}"
                f"&x={x}&y={y}&zoom=3"
            )
            try:
                response = requests.get(fallback_url)
                if response.status_code == 200:
                    tile = Image.open(io.BytesIO(response.content))
                    chosen_zoom = 3
                    return x, y, tile
            except Exception as e:
                print(e)

        return x, y, None

    def _is_black_tile(tile):
        if tile is None:
            return True
        tile_array = np.array(tile)
        return np.all(tile_array == 0)

    def _find_panorama_dimensions():
        tiles_cache = {}
        x, y = 5, 2

        is_first = True

        while True:
            tile_info = _fetch_tile(x, y)
            if tile_info is None:
                return None
            tile = tile_info[2]

            if tile is None:
                return None  # Invalid panorama

            if is_first:
                is_first = False
                if _is_black_tile(tile):
                    return None  # Invalid panorama

            tiles_cache[(x, y)] = tile

            if _is_black_tile(tile):
                y = y - 1

                while True:
                    tile_info = _fetch_tile(x, y)
                    if tile_info is None:
                        return None
                    tile = tile_info[2]
                    tiles_cache[(x, y)] = tile

                    if _is_black_tile(tile):
                        return x - 1, y, tiles_cache

                    x += 1

            x += 1
            y += 1

    def _fetch_remaining_tiles(max_x, max_y, existing_tiles):
        tiles_cache = existing_tiles.copy()

        with ThreadPoolExecutor(max_workers=1000) as executor:
            futures = []
            for x in range(max_x + 1):
                for y in range(max_y + 1):
                    if (x, y) not in tiles_cache:
                        futures.append(executor.submit(_fetch_tile, x, y))

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    x, y, tile = result
                    if tile is not None:
                        tiles_cache[(x, y)] = tile

        return tiles_cache

    def _assemble_panorama(tiles, max_x, max_y):
        if not tiles:
            return None
        tile_size = list(tiles.values())[0].size[0]
        panorama = Image.new('RGB', (tile_size * (max_x + 1), tile_size * (max_y + 1)))

        for (x, y), tile in tiles.items():
            panorama.paste(tile, (x * tile_size, y * tile_size))

        return panorama

    def _crop(image):
        img_array = np.array(image)
        y_nonzero, x_nonzero, _ = np.nonzero(img_array)
        if y_nonzero.size > 0 and x_nonzero.size > 0:
            return img_array[np.min(y_nonzero):np.max(y_nonzero) + 1, np.min(x_nonzero):np.max(x_nonzero) + 1]
        return img_array # Return original if all black

    dimension_result = _find_panorama_dimensions()
    if dimension_result is None:
        return None

    max_x, max_y, initial_tiles = dimension_result
    full_tiles = _fetch_remaining_tiles(max_x, max_y, initial_tiles)
    assembled_panorama = _assemble_panorama(full_tiles, max_x, max_y)
    if assembled_panorama is None:
        return None
    cropped_panorama = _crop(assembled_panorama)
    return cv2.cvtColor(cv2.resize(cropped_panorama, (8192, 4096), interpolation=cv2.INTER_LINEAR), cv2.COLOR_RGB2BGR)

def equirectangular_to_perspective(equi_img, fov, theta, phi, height, width):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = torch.tensor(equi_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    h, w = equi_img.shape[:2]

    hFOV = float(height) / width * fov
    w_len = torch.tan(torch.deg2rad(torch.tensor(fov / 2.0, device=device)))
    h_len = torch.tan(torch.deg2rad(torch.tensor(hFOV / 2.0, device=device)))

    x_map = torch.ones((height, width), dtype=torch.float32, device=device)
    y_map = torch.linspace(-w_len, w_len, width, device=device).repeat(height, 1)
    z_map = -torch.linspace(-h_len, h_len, height, device=device).unsqueeze(1).repeat(1, width)

    D = torch.sqrt(x_map**2 + y_map**2 + z_map**2)
    xyz = torch.stack((x_map, y_map, z_map), dim=-1) / D.unsqueeze(-1)

    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)

    R1, _ = cv2.Rodrigues((z_axis * torch.deg2rad(torch.tensor(theta))).cpu().numpy())
    R2, _ = cv2.Rodrigues((np.dot(R1, y_axis.cpu().numpy()) * -torch.deg2rad(torch.tensor(phi)).item()))

    R1 = torch.tensor(R1, dtype=torch.float32, device=device)
    R2 = torch.tensor(R2, dtype=torch.float32, device=device)

    xyz = xyz.view(-1, 3).T
    xyz = torch.matmul(R1, xyz)
    xyz = torch.matmul(R2, xyz).T
    xyz = xyz.view(height, width, 3)

    lat = torch.asin(xyz[:, :, 2])
    lon = torch.atan2(xyz[:, :, 1], xyz[:, :, 0])

    lon = lon / np.pi * (w - 1) / 2.0 + (w - 1) / 2.0
    lat = lat / (np.pi / 2.0) * (h - 1) / 2.0 + (h - 1) / 2.0

    lat = h - lat

    lon = (lon / ((w - 1) / 2.0)) - 1
    lat = (lat / ((h - 1) / 2.0)) - 1

    grid = torch.stack((lon, lat), dim=-1).unsqueeze(0)

    persp = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)

    return (persp[0].permute(1, 2, 0) * 255).byte().cpu().numpy()

def equirectangular_point_to_perspective(label_x, label_y, equi_width, equi_height, fov, theta, phi, height, width):
    """
    Convert a point from an equirectangular image (label_x, label_y) to its
    corresponding perspective projection pixel coordinates.
    
    Parameters:
      label_x, label_y : float
          Coordinates of the point in the equirectangular image.
      equi_width, equi_height : int or float
          Dimensions of the equirectangular image.
      fov : float
          Horizontal field-of-view (in degrees) for the perspective image.
      theta, phi : float
          Camera orientation angles (in degrees). Theta is the horizontal (azimuth) 
          and phi the vertical (elevation) angle.
      height, width : int or float
          Dimensions of the output perspective image.
    
    Returns:
      (x_img, y_img) : tuple of floats
          The (x, y) pixel coordinates in the perspective image, or None if the point is
          behind the camera.
    """
    # 1. Convert equirectangular pixel to spherical coordinates.
    # Map label_x from [0, equi_width] to longitude in degrees [-180, 180].
    lon_deg = (label_x / equi_width) * 360.0 - 180.0
    # Map label_y from [0, equi_height] to latitude in degrees [90, -90].
    lat_deg = 90.0 - (label_y / equi_height) * 180.0

    # Convert angles to radians.
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)

    # 2. Convert spherical (lat, lon) to 3D Cartesian coordinates on unit sphere.
    point = np.array([
        np.cos(lat) * np.cos(lon),
        np.sin(lat),
        np.cos(lat) * np.sin(lon)
    ])

    # 3. Define camera orientation.
    # Convert camera angles to radians.
    theta_rad = np.deg2rad(theta)
    phi_rad   = np.deg2rad(phi)
    # Camera's forward (look) vector.
    forward = np.array([
        np.cos(phi_rad) * np.cos(theta_rad),
        np.sin(phi_rad),
        np.cos(phi_rad) * np.sin(theta_rad)
    ])
    forward /= np.linalg.norm(forward)
    
    # Build a right-handed camera coordinate system.
    world_up = np.array([0, 1, 0])
    # Right vector: note the order of cross product matters.
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # If forward is nearly parallel to world_up, choose an alternate up vector.
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    # Recompute the true up vector.
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    
    # 4. Transform point to camera coordinate system.
    # p_cam_x, p_cam_y, p_cam_z are the coordinates along right, up, and forward.
    p_cam_x = np.dot(point, right)
    p_cam_y = np.dot(point, up)
    p_cam_z = np.dot(point, forward)
    
    # If the point is behind the camera, it cannot be projected.
    if p_cam_z <= 0:
        return None

    # 5. Compute the perspective projection.
    # Compute focal length from the horizontal field-of-view.
    # (width/2) = f * tan(fov/2)  =>  f = (width/2) / tan(fov/2)
    fov_rad = np.deg2rad(fov)
    f = (width / 2) / np.tan(fov_rad / 2)

    # Perspective projection (pinhole model):
    # x_img = (p_cam_x * f / p_cam_z) + (width / 2)
    # y_img = -(p_cam_y * f / p_cam_z) + (height / 2)
    # (The minus sign on y_img converts from a y-up to a y-down image coordinate system.)
    x_img = (p_cam_x * f / p_cam_z) + (width / 2)
    y_img = -(p_cam_y * f / p_cam_z) + (height / 2)
    
    return int(x_img), int(y_img)

def create_perspective_crop(equi_img, label_x_norm, label_y_norm):
    fov = 90
    height = 2048
    width = 2048
    equi_h, equi_w = equi_img.shape[:2]
    theta = (label_x_norm * equi_w / equi_w) * 360 - 180
    phi = 0

    perspective_img = equirectangular_to_perspective(equi_img, fov, theta, phi, height, width)
    img_h, img_w = perspective_img.shape[:2]

    persp_coords = equirectangular_point_to_perspective(
        label_x_norm * equi_w,
        label_y_norm * equi_h,
        equi_w,
        equi_h,
        fov,
        theta,
        phi,
        img_h,
        img_w
    )

    if persp_coords is None:
        return None

    center_x, center_y = persp_coords

    depth = model.infer_image(perspective_img)
    inv_depth = 1 / depth[center_y, center_x]
    crop_size_half = int(inv_depth * 6100)

    start_x = max(0, center_x - crop_size_half)
    start_y = max(0, center_y - crop_size_half)
    end_x = min(img_w, center_x + crop_size_half)
    end_y = min(img_h, center_y + crop_size_half)

    crop_height, crop_width = perspective_img[start_y:end_y, start_x:end_x].shape[:2]
    if crop_width > crop_height:
        resize_width, resize_height = 640, int(640 * crop_height / crop_width)
    else:
        resize_width, resize_height = int(640 * crop_width / crop_height), 640
    cropped_img = cv2.resize(perspective_img[start_y:end_y, start_x:end_x], (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
    return cropped_img

os.makedirs(f"{label_type}/correct", exist_ok=True)
os.makedirs(f"{label_type}/incorrect", exist_ok=True)

total_images = len(combined_filtered)
start_time = time.time()
processed_count = 0

for index, row in combined_filtered.iterrows():
    pano_id = row['gsv_panorama_id']
    label_class = row['label_class']
    label_x = row['pano_x']
    label_y = row['pano_y']
    pano_width = row['pano_width']
    pano_height = row['pano_height']

    equi_img_bgr = fetch_panorama(pano_id)

    if equi_img_bgr is not None:
        equi_img_rgb = cv2.cvtColor(equi_img_bgr, cv2.COLOR_BGR2RGB)
        equi_h, equi_w = equi_img_rgb.shape[:2]
        label_x_norm = label_x / pano_width
        label_y_norm = label_y / pano_height
        try:
            cropped_perspective = create_perspective_crop(equi_img_rgb, label_x_norm, label_y_norm)
            if cropped_perspective is not None and cropped_perspective.size > 0:
                filename = f"{label_type}/{label_class}/{row['city']}_{row['label_id']}.webp"
                cv2.imwrite(filename, cv2.cvtColor(cropped_perspective, cv2.COLOR_RGB2BGR))
                processed_count += 1
                elapsed_time = time.time() - start_time
                avg_time_per_image = elapsed_time / processed_count
                remaining_images = total_images - processed_count
                estimated_remaining_time = avg_time_per_image * remaining_images
                print(f"Processed {processed_count}/{total_images} images. Saved image to {filename}. "
                      f"Elapsed Time: {timedelta(seconds=int(elapsed_time))}, "
                      f"Time Left: {estimated_remaining_time}")
            else:
                processed_count += 1
                print(f"Processed {processed_count}/{total_images} images. Could not create a valid perspective crop for label ID {row['label_id']}.")
        except Exception as e:
            processed_count += 1
            print(f"Processed {processed_count}/{total_images} images. Error processing label ID {row['label_id']}: {e}")
    else:
        processed_count += 1
        print(f"Processed {processed_count}/{total_images} images. Could not fetch panorama for label ID {row['label_id']}")

print("Processing complete.")
