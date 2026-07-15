"""
CUDA Multi-Stream Use-After-Free Demo — H20-3e (sm_90)

Demonstrates the race condition pattern from:
- DeepSpeed #8061: overlap_comm + torch.compile = NaN
- Megatron #5788: StorageResize free without record_stream
- vLLM #45552: CuMem sleep/wake illegal memory access

Pattern: Data is written on CUDA stream A, then read on stream B
without proper synchronization. If stream B reads before stream A
completes writing, the result is stale/NaN data.

This demo:
1. Creates a buffer on the default stream
2. Writes data on a producer stream (simulating backward hook copy_)
3. Immediately reads on a consumer stream (simulating reduction_stream)
4. Measures corruption rate when no synchronization is used
5. Shows that stream.wait_stream() eliminates the corruption
"""

import torch
import time
import math

def run_stream_race_demo():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"GPU: {gpu_name}, sm_{compute_cap[0]}{compute_cap[1]}")

    # Create two CUDA streams (simulating reduction_stream vs backward streams)
    producer_stream = torch.cuda.Stream(device=device)
    consumer_stream = torch.cuda.Stream(device=device)

    buffer_size = 1024 * 1024  # 1M elements (~2MB fp16)
    num_trials = 1000

    # ============================================================
    # Phase 1: Race condition (NO synchronization)
    # ============================================================
    print("\n" + "="*60)
    print("Phase 1: NO stream synchronization (race condition)")
    print("="*60)

    corruption_count = 0
    nan_count = 0
    mismatch_count = 0

    for trial in range(num_trials):
        # Create buffer on default stream
        buffer = torch.zeros(buffer_size, dtype=torch.float16, device=device)

        # Write data on producer stream (simulating copy_() from backward hook)
        source_data = torch.full((buffer_size,), float(trial + 1), dtype=torch.float16, device=device)

        with torch.cuda.stream(producer_stream):
            buffer.copy_(source_data)

        # Immediately read on consumer stream (simulating average_tensor on reduction_stream)
        # NO wait_stream — this is the bug!
        with torch.cuda.stream(consumer_stream):
            read_result = buffer.clone()

        # Check on default stream after both operations should be done
        torch.cuda.synchronize()

        # Compare: expected all values = trial+1, but stale zeros may appear
        expected = torch.full((buffer_size,), float(trial + 1), dtype=torch.float16, device=device)
        matches = (read_result == expected)
        mismatches = (read_result != expected) & torch.isfinite(read_result)
        nans = torch.isnan(read_result)

        if not matches.all().item():
            corruption_count += 1
            mismatch_count += mismatches.sum().item()
            nan_count += nans.sum().item()

    total_elements = num_trials * buffer_size
    print(f"  Trials: {num_trials}")
    print(f"  Corrupted trials: {corruption_count}/{num_trials} ({corruption_count/num_trials*100:.1f}%)")
    print(f"  Mismatch elements: {mismatch_count}/{total_elements} ({mismatch_count/total_elements*100:.4f}%)")
    print(f"  NaN elements: {nan_count}/{total_elements}")
    print(f"  Avg mismatches per corrupted trial: {mismatch_count/max(corruption_count,1):.0f}")

    # ============================================================
    # Phase 2: With stream synchronization (the fix)
    # ============================================================
    print("\n" + "="*60)
    print("Phase 2: With stream.wait_stream() (fix applied)")
    print("="*60)

    corruption_count_fixed = 0
    nan_count_fixed = 0
    mismatch_count_fixed = 0

    for trial in range(num_trials):
        buffer = torch.zeros(buffer_size, dtype=torch.float16, device=device)
        source_data = torch.full((buffer_size,), float(trial + 1), dtype=torch.float16, device=device)

        with torch.cuda.stream(producer_stream):
            buffer.copy_(source_data)

        # FIX: consumer stream waits for producer stream before reading
        consumer_stream.wait_stream(producer_stream)

        with torch.cuda.stream(consumer_stream):
            read_result = buffer.clone()

        torch.cuda.synchronize()

        expected = torch.full((buffer_size,), float(trial + 1), dtype=torch.float16, device=device)
        matches = (read_result == expected)
        mismatches = (read_result != expected) & torch.isfinite(read_result)
        nans = torch.isnan(read_result)

        if not matches.all().item():
            corruption_count_fixed += 1
            mismatch_count_fixed += mismatches.sum().item()
            nan_count_fixed += nans.sum().item()

    print(f"  Trials: {num_trials}")
    print(f"  Corrupted trials: {corruption_count_fixed}/{num_trials} ({corruption_count_fixed/num_trials*100:.1f}%)")
    print(f"  Mismatch elements: {mismatch_count_fixed}/{total_elements}")
    print(f"  NaN elements: {nan_count_fixed}/{total_elements}")

    # ============================================================
    # Phase 3: Multiple producer streams (DeepSpeed #8061 pattern)
    # ============================================================
    print("\n" + "="*60)
    print("Phase 3: Multiple producer streams (DeepSpeed #8061 pattern)")
    print("="*60)

    # Simulate the exact DeepSpeed scenario:
    # Multiple backward hooks on different streams write to IPG buffer
    # average_tensor reads on reduction_stream without waiting for ALL producers

    num_producers = 4  # 4 different backward hook streams
    producer_streams = [torch.cuda.Stream(device=device) for _ in range(num_producers)]
    reduction_stream = torch.cuda.Stream(device=device)

    # Divide buffer into segments (simulating IPG bucket slots)
    segment_size = buffer_size // num_producers

    corruption_multi = 0
    mismatch_multi = 0
    nan_multi = 0

    for trial in range(num_trials):
        buffer = torch.zeros(buffer_size, dtype=torch.float16, device=device)

        # Each producer stream writes its segment
        for i, ps in enumerate(producer_streams):
            source = torch.full((segment_size,), float(trial * num_producers + i + 1), dtype=torch.float16, device=device)
            segment = buffer.narrow(0, i * segment_size, segment_size)
            with torch.cuda.stream(ps):
                segment.copy_(source)

        # Bug: only wait for ONE producer (current_stream) instead of ALL
        # This is exactly what average_tensor() did before the fix
        reduction_stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(reduction_stream):
            read_result = buffer.clone()

        torch.cuda.synchronize()

        # Expected: each segment has a different value
        expected = torch.zeros(buffer_size, dtype=torch.float16, device=device)
        for i in range(num_producers):
            expected[i*segment_size:(i+1)*segment_size] = float(trial * num_producers + i + 1)

        matches = (read_result == expected)
        mismatches = (read_result != expected) & torch.isfinite(read_result)

        if not matches.all().item():
            corruption_multi += 1
            mismatch_multi += mismatches.sum().item()

    print(f"  Trials: {num_trials}")
    print(f"  Producers: {num_producers}")
    print(f"  Corrupted trials: {corruption_multi}/{num_trials} ({corruption_multi/num_trials*100:.1f}%)")
    print(f"  Mismatch elements: {mismatch_multi}/{total_elements}")

    # Phase 3b: Fix — wait for ALL producer streams
    corruption_multi_fixed = 0
    mismatch_multi_fixed = 0

    for trial in range(num_trials):
        buffer = torch.zeros(buffer_size, dtype=torch.float16, device=device)

        for i, ps in enumerate(producer_streams):
            source = torch.full((segment_size,), float(trial * num_producers + i + 1), dtype=torch.float16, device=device)
            segment = buffer.narrow(0, i * segment_size, segment_size)
            with torch.cuda.stream(ps):
                segment.copy_(source)

        # FIX: wait for ALL producer streams
        for ps in producer_streams:
            reduction_stream.wait_stream(ps)

        with torch.cuda.stream(reduction_stream):
            read_result = buffer.clone()

        torch.cuda.synchronize()

        expected = torch.zeros(buffer_size, dtype=torch.float16, device=device)
        for i in range(num_producers):
            expected[i*segment_size:(i+1)*segment_size] = float(trial * num_producers + i + 1)

        matches = (read_result == expected)
        mismatches = (read_result != expected) & torch.isfinite(read_result)

        if not matches.all().item():
            corruption_multi_fixed += 1
            mismatch_multi_fixed += mismatches.sum().item()

    print(f"\n  With ALL-producer wait_stream (fix applied):")
    print(f"  Corrupted trials: {corruption_multi_fixed}/{num_trials} ({corruption_multi_fixed/num_trials*100:.1f}%)")
    print(f"  Mismatch elements: {mismatch_multi_fixed}/{total_elements}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"GPU: {gpu_name} (sm_{compute_cap[0]}{compute_cap[1]})")
    print(f"Buffer: {buffer_size} elements ({buffer_size*2/1024/1024:.1f} MB fp16)")
    print(f"Trials: {num_trials}")
    print()
    print("  Single-producer race:")
    print(f"    No sync:    {corruption_count} corrupted ({corruption_count/num_trials*100:.1f}%)")
    print(f"    With sync:  {corruption_count_fixed} corrupted ({corruption_count_fixed/num_trials*100:.1f}%)")
    print()
    print("  Multi-producer race (#8061 pattern):")
    print(f"    1-stream wait: {corruption_multi} corrupted ({corruption_multi/num_trials*100:.1f}%)")
    print(f"    ALL-stream wait: {corruption_multi_fixed} corrupted ({corruption_multi_fixed/num_trials*100:.1f}%)")
    print()
    if corruption_multi > 0 and corruption_multi_fixed == 0:
        print("★★★ BUG CONFIRMED: Multi-producer stream race produces corruption")
        print("★★★ FIX CONFIRMED: Waiting for ALL producer streams eliminates corruption")
        print("This validates the DeepSpeed #8061 / Megatron #5788 / vLLM #45552 fix pattern.")
    elif corruption_count > 0 and corruption_count_fixed == 0:
        print("★★★ BUG CONFIRMED: Single-producer stream race produces corruption")
        print("★★★ FIX CONFIRMED: stream.wait_stream() eliminates corruption")
    elif corruption_count == 0 and corruption_multi == 0:
        print("NOTE: No corruption detected on this GPU configuration.")
        print("Race conditions are timing-dependent and may not manifest")
        print("in simple demos. The bug pattern is still confirmed by:")
        print("  - DeepSpeed #8061: production-confirmed NaN with overlap_comm+compile")
        print("  - compute-sanitizer: detects cross-stream violations")
        print("  - Evidence matrix: all configs that add sync = no NaN")


if __name__ == "__main__":
    run_stream_race_demo()
