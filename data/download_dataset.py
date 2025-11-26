"""Dataset download script for NeRF Blender datasets."""

import os
import urllib.request
import zipfile
import argparse
from tqdm import tqdm


DATASET_URLS = {
    'lego': 'http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip',
    'all': 'http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_synthetic.zip'
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_zip(zip_path: str, extract_dir: str):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")


def download_nerf_dataset(scene: str = 'lego', output_dir: str = 'data'):
    os.makedirs(output_dir, exist_ok=True)
    
    if scene == 'lego':
        url = DATASET_URLS['lego']
        zip_name = 'nerf_example_data.zip'
    else:
        print("Downloading all scenes (this may take a while)...")
        url = DATASET_URLS['all']
        zip_name = 'nerf_synthetic.zip'
    
    zip_path = os.path.join(output_dir, zip_name)
    
    if os.path.exists(zip_path):
        print(f"ZIP file already exists: {zip_path}")
    else:
        print(f"Downloading {scene} dataset from {url}...")
        download_url(url, zip_path)
    
    extract_zip(zip_path, output_dir)
    
    print(f"Removing ZIP file...")
    os.remove(zip_path)
    
    print()
    print("=" * 60)
    print("Download complete!")
    
    if scene == 'lego':
        print(f"Dataset location: {os.path.join(output_dir, 'nerf_synthetic', 'lego')}")
        print()
        print("To train:")
        print(f"  python train.py --data_dir {os.path.join(output_dir, 'nerf_synthetic', 'lego')}")
    else:
        print(f"Datasets location: {os.path.join(output_dir, 'nerf_synthetic')}")
        print()
        print("Available scenes:", os.listdir(os.path.join(output_dir, 'nerf_synthetic')))
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Download NeRF datasets')
    parser.add_argument('--scene', type=str, default='lego', 
                       choices=['lego', 'all', 'ship', 'drums', 'ficus', 'hotdog', 'materials', 'mic', 'chair'],
                       help='Scene to download')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    download_nerf_dataset(args.scene, args.output_dir)


if __name__ == "__main__":
    main()
