#!/usr/bin/env python3
"""Apply TRL #6274 fix: Grid-aware multimodal field slicing in _tool_call_loop.

Bug: For Qwen-style VLMs, pixel_values is a flat [total_patches, C] tensor
indexed by image-patch, not by batch sample. Plain [v[i] for i in idxs_with_tool]
grabs ONE PATCH ROW per sample instead of ALL patches.

Fix: Add grid-aware slicing that computes per-sample patch spans from
image_grid_thw x merge_size^2 for correct pixel_values and image_grid_thw slicing.
"""

import os
import sys
import shutil

GRPO_TRAINER_PATH = "/jiangdingfeng/miniconda3/envs/env-flex/lib/python3.12/site-packages/trl/trainer/grpo_trainer.py"


def apply_fix(dry_run=False):
    with open(GRPO_TRAINER_PATH) as f:
        content = f.read()

    # Check if fix already applied
    if "_has_grid" in content:
        print("Fix already applied, skipping")
        return False

    # Find the target block to modify
    # We need to:
    # 1. Add grid-aware slicing setup after loop_images
    # 2. Replace `selected = [v[i] for i in idxs_with_tool]` with grid-aware logic

    # Step 1: Add _img_cnt and _has_grid setup after loop_images line
    target_loop_images = "            loop_images = [merged_images[i] for i in idxs_with_tool] if merged_images else None"
    if target_loop_images not in content:
        print("ERROR: Cannot find loop_images line")
        return False

    grid_setup = """            loop_images = [merged_images[i] for i in idxs_with_tool] if merged_images else None

            # For Qwen-style VLMs, pixel_values is a flat [total_patches, C] tensor indexed by
            # image-patch, not by batch sample. Plain [v[i] for i in idxs_with_tool] would grab
            # one PATCH row per sample (wrong) instead of all patches per sample. Compute per-sample
            # spans from image_grid_thw x merge_size^2 for correct slicing.
            _img_cnt = [len(imgs) if imgs else 0 for imgs in images] if images else []
            _has_grid = (
                _img_cnt
                and "pixel_values" in multimodal_fields
                and "image_grid_thw" in multimodal_fields
                and isinstance(multimodal_fields["pixel_values"], torch.Tensor)
            )
            if _has_grid:
                _thw = multimodal_fields["image_grid_thw"]
                _ms = getattr(self.processing_class.image_processor, "merge_size", 2)
                _ppi = (_thw[:, 0] * _thw[:, 1] * _ms ** 2).tolist()  # patches per image
                _img_off = 0
                _sample_spans = []  # (p_start, p_end, t_start, t_end) per batch sample
                for n in _img_cnt:
                    p_start = sum(_ppi[:_img_off])
                    p_end = sum(_ppi[:_img_off + n])
                    _sample_spans.append((p_start, p_end, _img_off, _img_off + n))
                    _img_off += n"""

    content = content.replace(target_loop_images, grid_setup, 1)

    # Step 2: Replace the plain slicing with grid-aware logic
    # Target: selected = [v[i] for i in idxs_with_tool]
    # This is inside the multimodal_fields loop

    target_selected = "                    selected = [v[i] for i in idxs_with_tool]"
    if target_selected not in content:
        print("ERROR: Cannot find selected slicing line")
        return False

    grid_slicing = """                    if _has_grid and k == "pixel_values":
                        # Image-patch-indexed: gather patches by sample span
                        _slices = []
                        for src in idxs_with_tool:
                            p_start, p_end, _, _ = _sample_spans[src]
                            _slices.append(v[p_start:p_end])
                        selected = [torch.cat(_slices, dim=0)] if _slices else [v.new_zeros(0, v.size(1))]
                    elif _has_grid and k in ("image_grid_thw", "pixel_attention_mask"):
                        # Image-indexed: gather entries by sample span
                        _slices = []
                        for src in idxs_with_tool:
                            _, _, t_start, t_end = _sample_spans[src]
                            _slices.append(v[t_start:t_end])
                        selected = [torch.cat(_slices, dim=0)] if _slices else [v.new_zeros(0, *v.shape[1:])]
                    else:
                        # Batch-indexed fields (token_type_ids, mm_token_type_ids, etc.)
                        selected = [v[i] for i in idxs_with_tool]"""

    content = content.replace(target_selected, grid_slicing, 1)

    if not dry_run:
        backup_path = GRPO_TRAINER_PATH + ".6274.orig"
        if not os.path.exists(backup_path):
            shutil.copy2(GRPO_TRAINER_PATH, backup_path)
            print("Backup saved to: %s" % backup_path)

        with open(GRPO_TRAINER_PATH, 'w') as f:
            f.write(content)
        print("Fixed file written")

        # Verify
        with open(GRPO_TRAINER_PATH) as f:
            verify = f.read()
        if "_has_grid" in verify and "_sample_spans" in verify:
            print("Fix verified successfully")
        else:
            print("ERROR: Fix verification failed")
            shutil.copy2(backup_path, GRPO_TRAINER_PATH)
            return False

    return True


def verify():
    """Verify the #6274 fix."""
    print("\n=== #6274 Verification ===")
    with open(GRPO_TRAINER_PATH) as f:
        content = f.read()

    has_grid_setup = "_has_grid" in content
    has_spans = "_sample_spans" in content
    has_pixel_slicing = "pixel_values" in content and "p_start" in content
    has_grid_thw_slicing = "image_grid_thw" in content and "t_start" in content
    has_fallback = "Batch-indexed fields" in content

    print("  Grid setup (_has_grid): %s" % has_grid_setup)
    print("  Sample spans (_sample_spans): %s" % has_spans)
    print("  pixel_values grid slicing: %s" % has_pixel_slicing)
    print("  image_grid_thw grid slicing: %s" % has_grid_thw_slicing)
    print("  Fallback batch slicing: %s" % has_fallback)

    if all([has_grid_setup, has_spans, has_pixel_slicing, has_grid_thw_slicing, has_fallback]):
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED")

    # Show the key sections
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "_has_grid =" in line or "_has_grid and" in line:
            print("  L%d: %s" % (i+1, line.strip()))


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    apply_fix(dry_run=dry_run)
    verify()
