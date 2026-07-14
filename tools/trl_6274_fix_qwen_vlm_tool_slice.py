"""
TRL #6274 Fix: Grid-aware multimodal field slicing in _tool_call_loop

## Root Cause

In `_tool_call_loop` (trl/trainer/grpo_trainer.py, line 1928-1940), multimodal
fields are subsetted for tool-calling samples using a plain batch-index slice:

```python
selected = [v[i] for i in idxs_with_tool]  # line 1931
```

For Qwen-style VLMs, `pixel_values` is NOT indexed by batch sample. It is a
flat 2D tensor `[total_patches, feature_dim]` where patches from ALL images
across ALL batch samples are concatenated. The per-image patch count is
determined by `image_grid_thw`.

The plain slice `v[i]` grabs ONE PATCH ROW per sample instead of ALL patches
belonging to that sample. When there are 560 total patches and batch_size=2,
`v[i] for i in [0,1]` selects only 2 rows. The vision tower expects 560 →
crash.

## Fix

Replace the plain slice with grid-aware slicing that:
1. Computes per-sample patch spans from `image_grid_thw` × `merge_size²`
2. Uses image counts from the original `images` parameter (unchanged during
   tool loop iterations)
3. Applies correct `[start:end]` slicing for pixel_values and image_grid_thw
4. Falls back to existing plain slicing for batch-indexed fields
   (token_type_ids, mm_token_type_ids, etc.)

## Unified Diff

```diff
--- a/trl/trainer/grpo_trainer.py
+++ b/trl/trainer/grpo_trainer.py
@@ -1925,21 +1925,53 @@ class GRPOTrainer(_BaseTrainer):
             if not idxs_with_tool:
                 break  # all overlong, exit tool loop

-            # Filter images and multimodal fields to match the current subset (index into full batch).
-            # Merge tool response images so the model can see visual feedback during generation.
+            # Filter images and multimodal fields to match the current subset.
+            # Merge tool response images for visual feedback during generation.
             merged_images = images
             if any(imgs for imgs in tool_images):
                 if merged_images is None:
                     merged_images = [imgs if imgs else None for imgs in tool_images]
                 else:
                     merged_images = [
                         (existing or []) + new for existing, new in zip(merged_images, tool_images, strict=True)
                     ]
             loop_images = [merged_images[i] for i in idxs_with_tool] if merged_images else None
+
+            # For Qwen-style VLMs, pixel_values is a flat [total_patches, C] tensor indexed by
+            # image-patch, not by batch sample. Plain [v[i] for i in idxs_with_tool] would grab
+            # one PATCH row per sample (wrong) instead of all patches per sample. Compute per-sample
+            # spans from image_grid_thw × merge_size² for correct slicing.
+            _img_cnt = [len(imgs) if imgs else 0 for imgs in images] if images else []
+            _has_grid = (
+                _img_cnt
+                and "pixel_values" in multimodal_fields
+                and "image_grid_thw" in multimodal_fields
+                and isinstance(multimodal_fields["pixel_values"], torch.Tensor)
+            )
+            if _has_grid:
+                _thw = multimodal_fields["image_grid_thw"]
+                _ms = getattr(self.processing_class.image_processor, "merge_size", 2)
+                _ppi = (_thw[:, 0] * _thw[:, 1] * _ms ** 2).tolist()  # patches per image
+                _img_off = 0
+                _sample_spans = []  # (p_start, p_end, t_start, t_end) per batch sample
+                for n in _img_cnt:
+                    p_start = sum(_ppi[:_img_off])
+                    p_end = sum(_ppi[:_img_off + n])
+                    _sample_spans.append((p_start, p_end, _img_off, _img_off + n))
+                    _img_off += n
+
             if multimodal_fields:
                 loop_multimodal_fields = {}
                 for k, v in multimodal_fields.items():
-                    selected = [v[i] for i in idxs_with_tool]
+                    if _has_grid and k == "pixel_values":
+                        # Image-patch-indexed: gather patches by sample span
+                        _slices = []
+                        for src in idxs_with_tool:
+                            p_start, p_end, _, _ = _sample_spans[src]
+                            _slices.append(v[p_start:p_end])
+                        selected = [torch.cat(_slices, dim=0)] if _slices else [v.new_zeros(0, v.size(1))]
+                    elif _has_grid and k in ("image_grid_thw", "pixel_attention_mask"):
+                        # Image-indexed: gather entries by sample span
+                        _slices = []
+                        for src in idxs_with_tool:
+                            _, _, t_start, t_end = _sample_spans[src]
+                            _slices.append(v[t_start:t_end])
+                        selected = [torch.cat(_slices, dim=0)] if _slices else [v.new_zeros(0, *v.shape[1:])]
+                    else:
+                        # Batch-indexed fields (token_type_ids, mm_token_type_ids, etc.)
+                        selected = [v[i] for i in idxs_with_tool]
                     # Per-token fields (e.g. token_type_ids) need zero-padding to match extended prompt length
                     if isinstance(selected[0], list):
                         selected = [

== Explanation ==

1. _img_cnt counts original images per sample from the `images` parameter
   (which is NOT modified during tool loop iterations — tool_images are added
   to a local `merged_images` variable, not to `images`).

2. _has_grid checks if grid-aware slicing is needed (Qwen-style VLMs).

3. _ppi computes patches per image from image_grid_thw × merge_size².

4. _sample_spans maps each batch sample to its (pixel_start, pixel_end,
   thw_start, thw_end) in the flat tensors.

5. For pixel_values: slice [p_start:p_end] per tool-calling sample and
   concatenate. This selects ALL patches belonging to each sample.

6. For image_grid_thw/pixel_attention_mask: slice [t_start:t_end] per
   tool-calling sample and concatenate.

7. For all other fields (token_type_ids, mm_token_type_ids, etc.): use the
   existing plain batch-index slice which is correct.

== Testing ==

Run the GRPO multi-turn VLM training repro from issue #6274. Before the fix:
vision tower crashes with dimension mismatch. After the fix: pixel_values has
the correct number of patches for the selected subset of samples.

== Total LOC ==
~30 lines added, ~2 lines modified
"""
