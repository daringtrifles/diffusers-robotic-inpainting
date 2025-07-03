#!/usr/bin/env python3
"""
Chunked in-painting runner for E2FGVI-HQ

Processes long sequences in 200-frame chunks to avoid OOM,
and preserves original frame names in the final output directory.
"""
import cv2
import os
import glob
import shutil
import tempfile
import subprocess
from typing import List, Optional, Tuple
import argparse




CKPT_PATH = "release_model/E2FGVI-HQ-CVPR22.pth"

GPU_ID = 7
CHUNK_SIZE = 200
WIDTH, HEIGHT = None, None
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
# ─────────────────────────────────────────────────────────────────────────


def discover_pairs(start,end, dir_root)-> List[Tuple[str, str, str]]:
    """Find (video_dir, mask_dir, save_dir) needing processing."""


    pairs = []
    for i in range(start, end + 1):
        video_dir = os.path.join(dir_root, f"{i}", "frames")

        mask_dir = os.path.join(dir_root, f"{i}", "mask_frames")
        save_dir = os.path.join(dir_root, f"{i}", "inpainted_frames")
        pairs.append((video_dir, mask_dir, save_dir))
    return pairs

def get_common_frames(video_dir: str, mask_dir: str) -> List[str]:

    v_files = {f for f in os.listdir(video_dir) if f.lower().endswith(IMG_EXTS)}
    m_files = {f for f in os.listdir(mask_dir) if f.lower().endswith(IMG_EXTS)}
    return sorted(v_files & m_files)


def create_symlink_chunk(src_dir: str, filenames: List[str]) -> str:
    tmpdir = tempfile.mkdtemp(prefix="chunk_")
    for i, fname in enumerate(filenames):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(tmpdir, f"{i:06d}" + os.path.splitext(fname)[-1])  # 000000.jpg
        os.symlink(src, dst)
    return tmpdir


def run_chunk(v_chunk: str, m_chunk: str, output_dir: str) -> None:
    cmd = [
        "python", "demo.py",
        "--model", "e2fgvi_hq",
        "--video", v_chunk,
        "--mask",  m_chunk,
        "--ckpt",  CKPT_PATH,
        "--set_size",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
        "--save_frame", output_dir
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def run_episode(video_dir: str, mask_dir: str, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    all_frames = get_common_frames(video_dir, mask_dir)

    chunks = [all_frames[i:i + CHUNK_SIZE]
              for i in range(0, len(all_frames), CHUNK_SIZE)]
    
    for i, chunk_filenames in enumerate(chunks):
        print(f"[INFO] Processing chunk {i + 1}/{len(chunks)}")

        v_tmp = create_symlink_chunk(video_dir, chunk_filenames)
        m_tmp = create_symlink_chunk(mask_dir, chunk_filenames)
        output_tmp = tempfile.mkdtemp(prefix="out_chunk_")

        try:
            run_chunk(v_tmp, m_tmp, output_tmp)

            # Move and rename outputs to final save_dir using original filenames
            for j, fname in enumerate(sorted(os.listdir(output_tmp))):
                if not fname.lower().endswith(IMG_EXTS):
                    continue
                src_path = os.path.join(output_tmp, fname)
                original_name = chunk_filenames[j]  # e.g., 000240.jpg
                dst_path = os.path.join(save_dir, original_name)
                shutil.move(src_path, dst_path)

        finally:
            shutil.rmtree(v_tmp, ignore_errors=True)
            shutil.rmtree(m_tmp, ignore_errors=True)
            shutil.rmtree(output_tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Chunked in-painting runner for E2FGVI-HQ")
    parser.add_argument('--dir_root', type=str, required=True, help='Root directory containing episode folders')
    parser.add_argument('--start', type=int, required=True, help='Start episode index (inclusive)')
    parser.add_argument('--end', type=int, required=True, help='End episode index (inclusive)')
    parser.add_argument('--width', type=int, required=True, help='Width of the mask frame')
    parser.add_argument('--height', type=int, required=True, help='Height of the mask frame')
    args = parser.parse_args()

    dir_root = args.dir_root
    start = args.start
    end = args.end
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = args.width, args.height
    pairs = discover_pairs(start, end, dir_root)
    print(f"[INFO] Found {len(pairs)} episode(s) to process.")

    for video_dir, mask_dir, save_dir in pairs:
        run_episode(video_dir, mask_dir, save_dir)


if __name__ == "__main__":
    main()