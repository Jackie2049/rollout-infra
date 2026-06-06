#!/usr/bin/env python3
"""PR #6 KV Transfer Guard Validation
====================================

Validates the correctness of replacing assertions with defensive guards
in vLLM's _update_from_kv_xfer_finished() method.

PR #6 changes:
1. finished_recving: assert req_id in self.requests → if guard + warning
2. finished_recving: assert RequestStatus.is_finished(req.status) → if guard + warning
3. finished_sending: assert req_id in self.requests → if guard + warning

This script tests 4 scenarios without needing a running vLLM instance:
1. Normal operation: req_id exists → blocks freed (same as before)
2. Race condition: req_id absent → guard skips, no crash
3. Unexpected status: req exists but wrong status → guard skips
4. Mixed: some reqs normal, some aborted → partial handling

Reference: Jackie2049/vllm PR #6, Issue #1
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
import json
import time

print("PR #6 KV Transfer Guard Validation")
print("=" * 60)


# ============================================================
# Simulate the scheduler state and guard behavior
# ============================================================

class RequestStatus:
    """Minimal mock of vLLM RequestStatus"""
    WAITING_FOR_REMOTE_KVS = "WAITING_FOR_REMOTE_KVS"
    FINISHED_ABORTED = "FINISHED_ABORTED"
    FINISHED_LENGTH_CAPPED = "FINISHED_LENGTH_CAPPED"

    @staticmethod
    def is_finished(status):
        return status in (RequestStatus.FINISHED_ABORTED,
                          RequestStatus.FINISHED_LENGTH_CAPPED)


class MockRequest:
    """Minimal mock of vLLM Request"""
    def __init__(self, req_id, status, num_blocks=5):
        self.req_id = req_id
        self.status = status
        self.num_blocks = num_blocks


class MockScheduler:
    """Simulates _update_from_kv_xfer_finished behavior with guards."""

    def __init__(self):
        self.requests: dict[str, MockRequest] = {}
        self.freed_blocks: list[str] = []
        self.finished_recving_kv_req_ids: set = set()
        self.warnings: list[str] = []

    def _free_blocks(self, req: MockRequest):
        """Record block freeing"""
        self.freed_blocks.append(req.req_id)

    def _log_warning(self, msg: str):
        self.warnings.append(msg)

    # --- OLD BEHAVIOR (assert, crashes on race) ---
    def update_from_kv_xfer_old(self, finished_recving, finished_sending):
        """Original code with assertions - crashes on race condition"""
        for req_id in finished_recving or []:
            assert req_id in self.requests  # CRASH if req aborted!
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)  # CRASH on unexpected status!
                self._free_blocks(req)

        for req_id in finished_sending or []:
            assert req_id in self.requests  # CRASH if req aborted!
            self._free_blocks(self.requests[req_id])

    # --- NEW BEHAVIOR (guard, gracefully handles race) ---
    def update_from_kv_xfer_new(self, finished_recving, finished_sending):
        """PR #6 code with defensive guards - no crash on race"""
        for req_id in finished_recving or []:
            if req_id not in self.requests:
                self._log_warning(
                    f"Request {req_id} not found when finishing KV recv; "
                    "may have been aborted during transfer"
                )
                continue
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                if not RequestStatus.is_finished(req.status):
                    self._log_warning(
                        f"Request {req_id} in unexpected status {req.status} "
                        "when finishing KV recv"
                    )
                    continue
                self._free_blocks(req)

        for req_id in finished_sending or []:
            if req_id not in self.requests:
                self._log_warning(
                    f"Request {req_id} not found when finishing KV send; "
                    "may have been aborted during transfer"
                )
                continue
            self._free_blocks(self.requests[req_id])


# ============================================================
# Test Scenarios
# ============================================================

results = []

def test_normal_operation():
    """Scenario 1: All req_ids exist → same behavior as assert version"""
    print("\n1. Normal Operation: all req_ids present")

    # Old version
    s_old = MockScheduler()
    s_old.requests = {
        "req_A": MockRequest("req_A", RequestStatus.WAITING_FOR_REMOTE_KVS),
        "req_B": MockRequest("req_B", RequestStatus.FINISHED_LENGTH_CAPPED),
        "req_C": MockRequest("req_C", RequestStatus.WAITING_FOR_REMOTE_KVS),
    }
    s_old.update_from_kv_xfer_old(
        finished_recving=["req_A", "req_B"],
        finished_sending=["req_C"]
    )

    # New version
    s_new = MockScheduler()
    s_new.requests = {
        "req_A": MockRequest("req_A", RequestStatus.WAITING_FOR_REMOTE_KVS),
        "req_B": MockRequest("req_B", RequestStatus.FINISHED_LENGTH_CAPPED),
        "req_C": MockRequest("req_C", RequestStatus.WAITING_FOR_REMOTE_KVS),
    }
    s_new.update_from_kv_xfer_new(
        finished_recving=["req_A", "req_B"],
        finished_sending=["req_C"]
    )

    # Compare
    recv_match = s_old.finished_recving_kv_req_ids == s_new.finished_recving_kv_req_ids
    freed_match = s_old.freed_blocks == s_new.freed_blocks
    no_warnings = len(s_new.warnings) == 0

    passed = recv_match and freed_match and no_warnings
    print(f"   recv_ids match: {recv_match}")
    print(f"   freed_blocks match: {freed_match}")
    print(f"   no warnings: {no_warnings}")
    print(f"   {'PASS' if passed else 'FAIL'}")

    results.append({
        "test": "normal_operation",
        "recv_match": recv_match,
        "freed_match": freed_match,
        "no_warnings": no_warnings,
        "pass": passed,
    })
    return passed


def test_race_condition_recv():
    """Scenario 2: req_id absent in finished_recving → old crashes, new skips"""
    print("\n2. Race Condition (recv): req aborted before KV recv finished")

    # Old version → would crash with AssertionError
    s_old = MockScheduler()
    s_old.requests = {"req_A": MockRequest("req_A", RequestStatus.WAITING_FOR_REMOTE_KVS)}
    crashed = False
    try:
        s_old.update_from_kv_xfer_old(
            finished_recving=["req_A", "req_ABORTED"],  # req_ABORTED not in requests!
            finished_sending=[]
        )
    except AssertionError as e:
        crashed = True
        print(f"   Old version CRASHED: {e}")

    # New version → gracefully handles
    s_new = MockScheduler()
    s_new.requests = {"req_A": MockRequest("req_A", RequestStatus.WAITING_FOR_REMOTE_KVS)}
    s_new.update_from_kv_xfer_new(
        finished_recving=["req_A", "req_ABORTED"],
        finished_sending=[]
    )

    # req_A still processed correctly
    a_processed = "req_A" in s_new.finished_recving_kv_req_ids
    aborted_skipped = "req_ABORTED" not in s_new.finished_recving_kv_req_ids
    warning_logged = any("req_ABORTED" in w for w in s_new.warnings)

    passed = crashed and a_processed and aborted_skipped and warning_logged
    print(f"   Old version crashed (expected): {crashed}")
    print(f"   New version processed req_A: {a_processed}")
    print(f"   New version skipped req_ABORTED: {aborted_skipped}")
    print(f"   Warning logged: {warning_logged}")
    print(f"   {'PASS' if passed else 'FAIL'}")

    results.append({
        "test": "race_condition_recv",
        "old_crashed": crashed,
        "new_a_processed": a_processed,
        "new_aborted_skipped": aborted_skipped,
        "warning_logged": warning_logged,
        "pass": passed,
    })
    return passed


def test_race_condition_send():
    """Scenario 3: req_id absent in finished_sending → old crashes, new skips"""
    print("\n3. Race Condition (send): req aborted before KV send finished")

    # Old version → would crash
    s_old = MockScheduler()
    s_old.requests = {"req_A": MockRequest("req_A", RequestStatus.WAITING_FOR_REMOTE_KVS)}
    crashed = False
    try:
        s_old.update_from_kv_xfer_old(
            finished_recving=[],
            finished_sending=["req_ABORTED"]  # not in requests!
        )
    except AssertionError as e:
        crashed = True
        print(f"   Old version CRASHED: {e}")

    # New version → gracefully handles
    s_new = MockScheduler()
    s_new.requests = {"req_A": MockRequest("req_A", RequestStatus.WAITING_FOR_REMOTE_KVS)}
    s_new.update_from_kv_xfer_new(
        finished_recving=[],
        finished_sending=["req_ABORTED"]
    )

    no_freed = len(s_new.freed_blocks) == 0
    warning_logged = any("req_ABORTED" in w for w in s_new.warnings)

    passed = crashed and no_freed and warning_logged
    print(f"   Old version crashed (expected): {crashed}")
    print(f"   New version: no blocks freed: {no_freed}")
    print(f"   Warning logged: {warning_logged}")
    print(f"   {'PASS' if passed else 'FAIL'}")

    results.append({
        "test": "race_condition_send",
        "old_crashed": crashed,
        "new_no_freed": no_freed,
        "warning_logged": warning_logged,
        "pass": passed,
    })
    return passed


def test_unexpected_status():
    """Scenario 4: req exists but unexpected status → old crashes, new skips"""
    print("\n4. Unexpected Status: req in RUNNING status during KV recv finish")

    RUNNING = "RUNNING"  # Not a valid finished status

    # Old version → would crash on assert RequestStatus.is_finished()
    s_old = MockScheduler()
    s_old.requests = {"req_X": MockRequest("req_X", RUNNING)}
    crashed = False
    try:
        s_old.update_from_kv_xfer_old(
            finished_recving=["req_X"],
            finished_sending=[]
        )
    except AssertionError as e:
        crashed = True
        print(f"   Old version CRASHED: {e}")

    # New version → gracefully handles with warning
    s_new = MockScheduler()
    s_new.requests = {"req_X": MockRequest("req_X", RUNNING)}
    s_new.update_from_kv_xfer_new(
        finished_recving=["req_X"],
        finished_sending=[]
    )

    not_in_recv = "req_X" not in s_new.finished_recving_kv_req_ids
    not_freed = "req_X" not in s_new.freed_blocks
    warning_logged = any("unexpected status" in w for w in s_new.warnings)

    passed = crashed and not_in_recv and not_freed and warning_logged
    print(f"   Old version crashed (expected): {crashed}")
    print(f"   New version: req not in recv_ids: {not_in_recv}")
    print(f"   New version: blocks not freed: {not_freed}")
    print(f"   Warning logged: {warning_logged}")
    print(f"   {'PASS' if passed else 'FAIL'}")

    results.append({
        "test": "unexpected_status",
        "old_crashed": crashed,
        "new_not_in_recv": not_in_recv,
        "new_not_freed": not_freed,
        "warning_logged": warning_logged,
        "pass": passed,
    })
    return passed


def test_mixed_scenario():
    """Scenario 5: Mix of normal + aborted + unexpected status"""
    print("\n5. Mixed Scenario: 3 reqs normal, 2 aborted, 1 unexpected status")

    # New version only (old would crash)
    s = MockScheduler()
    s.requests = {
        "req_normal_recv": MockRequest("req_normal_recv", RequestStatus.WAITING_FOR_REMOTE_KVS),
        "req_normal_send": MockRequest("req_normal_send", RequestStatus.FINISHED_LENGTH_CAPPED),
        "req_finished_recv": MockRequest("req_finished_recv", RequestStatus.FINISHED_LENGTH_CAPPED),
    }
    # Note: req_abort_recv, req_abort_send, req_unexpected NOT in self.requests

    s.update_from_kv_xfer_new(
        finished_recving=["req_normal_recv", "req_abort_recv", "req_finished_recv",
                          "req_unexpected"],
        finished_sending=["req_normal_send", "req_abort_send"]
    )

    # Verify
    normal_recv_ok = "req_normal_recv" in s.finished_recving_kv_req_ids
    finished_recv_ok = "req_finished_recv" not in s.finished_recving_kv_req_ids  # is_finished → free blocks
    finished_recv_freed = "req_finished_recv" in s.freed_blocks
    normal_send_freed = "req_normal_send" in s.freed_blocks
    abort_recv_skipped = "req_abort_recv" not in s.finished_recving_kv_req_ids
    abort_send_skipped = "req_abort_send" not in s.freed_blocks
    warnings_count = len(s.warnings)  # Should be 3: abort_recv, abort_send, unexpected

    passed = (normal_recv_ok and finished_recv_ok and finished_recv_freed
              and normal_send_freed and abort_recv_skipped and abort_send_skipped
              and warnings_count >= 3)

    print(f"   Normal recv processed: {normal_recv_ok}")
    print(f"   Finished recv freed (not in recv_ids): {finished_recv_ok}")
    print(f"   Finished recv blocks freed: {finished_recv_freed}")
    print(f"   Normal send blocks freed: {normal_send_freed}")
    print(f"   Abort recv skipped: {abort_recv_skipped}")
    print(f"   Abort send skipped: {abort_send_skipped}")
    print(f"   Warnings logged: {warnings_count}")
    print(f"   {'PASS' if passed else 'FAIL'}")

    results.append({
        "test": "mixed_scenario",
        "normal_recv_ok": normal_recv_ok,
        "finished_recv_freed": finished_recv_freed,
        "normal_send_freed": normal_send_freed,
        "abort_recv_skipped": abort_recv_skipped,
        "abort_send_skipped": abort_send_skipped,
        "warnings_count": warnings_count,
        "pass": passed,
    })
    return passed


def test_empty_finished():
    """Scenario 6: Empty finished lists → no crash, no action"""
    print("\n6. Empty Finished Lists: no KV transfer events")

    s = MockScheduler()
    s.requests = {"req_A": MockRequest("req_A", RequestStatus.WAITING_FOR_REMOTE_KVS)}
    s.update_from_kv_xfer_new(
        finished_recving=None,  # vLLM uses `or ()` pattern
        finished_sending=None
    )

    no_recv = len(s.finished_recving_kv_req_ids) == 0
    no_freed = len(s.freed_blocks) == 0
    no_warnings = len(s.warnings) == 0

    passed = no_recv and no_freed and no_warnings
    print(f"   No recv_ids added: {no_recv}")
    print(f"   No blocks freed: {no_freed}")
    print(f"   No warnings: {no_warnings}")
    print(f"   {'PASS' if passed else 'FAIL'}")

    results.append({
        "test": "empty_finished",
        "no_recv": no_recv,
        "no_freed": no_freed,
        "no_warnings": no_warnings,
        "pass": passed,
    })
    return passed


# ============================================================
# Run All Tests
# ============================================================

all_pass = True
all_pass &= test_normal_operation()
all_pass &= test_race_condition_recv()
all_pass &= test_race_condition_send()
all_pass &= test_unexpected_status()
all_pass &= test_mixed_scenario()
all_pass &= test_empty_finished()

print("\n" + "=" * 60)
print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
print("=" * 60)

# Save results
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'kv_transfer_guard_test_results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {out_path}")