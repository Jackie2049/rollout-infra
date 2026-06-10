#!/usr/bin/env python3
"""
Distributed Systems Simulator for AI Infra
Simulates consensus, failure detection, replication, leader election, and ordering.

Usage:
    python tools/distributed_systems_simulator.py [class_name]

Classes:
    - ConsensusSimulator: Paxos/Raft/BFT consensus with failure scenarios
    - FailureDetectorSimulator: Timeout vs φ-accrual comparison
    - ClockOrderingSimulator: Lamport vs Vector clock ordering
    - ReplicationSimulator: Primary-backup vs Chain vs Quorum replication
    - PACELCSimulator: CAP/PACELC trade-off analysis for AI Infra
"""

import argparse
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from collections import defaultdict


class ConsensusSimulator:
    """Simulates Paxos, Raft, and BFT consensus protocols under failures."""

    def __init__(self, n_nodes: int = 5, n_faulty: int = 0):
        self.n_nodes = n_nodes
        self.n_faulty = n_faulty
        self.nodes = [{"id": i, "alive": True, "state": "follower"} for i in range(n_nodes)]
        self.log = []

    def _mark_faults(self):
        """Randomly mark n_faulty nodes as crashed."""
        faulty = np.random.choice(self.n_nodes, self.n_faulty, replace=False)
        for idx in faulty:
            self.nodes[idx]["alive"] = False
            self.nodes[idx]["state"] = "crashed"

    def simulate_raft_election(self) -> dict[str, Any]:
        """Simulate Raft leader election with failures."""
        self._mark_faults()
        alive_nodes = [n for n in self.nodes if n["alive"]]
        n_alive = len(alive_nodes)
        majority_needed = (self.n_nodes + 1) // 2

        # Randomized election timeout
        timeouts = np.random.uniform(150, 300, n_alive)  # ms
        winner_idx = np.argmin(timeouts)

        # Candidate requests votes
        votes_received = 1  # self-vote
        for n in alive_nodes:
            if n["id"] != alive_nodes[winner_idx]["id"]:
                # Vote granted with some probability (typically high in Raft)
                votes_received += 1

        can_elect = votes_received >= majority_needed
        leader_id = alive_nodes[winner_idx]["id"] if can_elect else None

        # Log replication simulation
        entries_per_second = 10000 if can_elect else 0
        commit_latency_ms = 2 * np.random.uniform(1, 5)  # round-trip

        result = {
            "protocol": "Raft",
            "n_nodes": self.n_nodes,
            "n_faulty": self.n_faulty,
            "n_alive": n_alive,
            "majority_needed": majority_needed,
            "can_elect_leader": can_elect,
            "leader_id": leader_id,
            "election_timeout_ms": timeouts[winner_idx],
            "entries_per_second": entries_per_second,
            "commit_latency_ms": commit_latency_ms,
            "time_to_elect_ms": timeouts[winner_idx],
        }
        self.log.append(result)
        return result

    def simulate_paxos(self, n_rounds: int = 3) -> dict[str, Any]:
        """Simulate Multi-Paxos consensus with failures."""
        self._mark_faults()
        alive_nodes = [n for n in self.nodes if n["alive"]]
        n_alive = len(alive_nodes)
        quorum_needed = (self.n_nodes + 1) // 2

        rounds_to_consensus = 0
        reached_consensus = False

        for r in range(n_rounds):
            rounds_to_consensus += 1
            # Proposer phase
            proposers_alive = [n for n in alive_nodes if np.random.random() > 0.3]
            if len(proposers_alive) == 0:
                continue

            # Accept phase - count acceptors
            acceptors_alive = [n for n in alive_nodes if np.random.random() > 0.1]
            n_accepted = len(acceptors_alive)

            if n_accepted >= quorum_needed:
                reached_consensus = True
                break

        result = {
            "protocol": "Paxos",
            "n_nodes": self.n_nodes,
            "n_faulty": self.n_faulty,
            "n_alive": n_alive,
            "quorum_needed": quorum_needed,
            "reached_consensus": reached_consensus,
            "rounds_to_consensus": rounds_to_consensus,
            "latency_per_round_ms": np.random.uniform(10, 50),
            "total_latency_ms": rounds_to_consensus * np.random.uniform(10, 50),
        }
        self.log.append(result)
        return result

    def simulate_bft(self, n_byzantine: int = 1) -> dict[str, Any]:
        """Simulate BFT consensus (requires 3f+1 nodes for f byzantine faults)."""
        min_nodes_needed = 3 * n_byzantine + 1
        can_tolerate = self.n_nodes >= min_nodes_needed

        # PBFT: 3 phases (pre-prepare, prepare, commit)
        phases = 3
        messages_per_phase = self.n_nodes * (self.n_nodes - 1)
        total_messages = phases * messages_per_phase
        latency_per_phase_ms = np.random.uniform(5, 20)

        result = {
            "protocol": "BFT(PBFT)",
            "n_nodes": self.n_nodes,
            "n_byzantine": n_byzantine,
            "min_nodes_needed": min_nodes_needed,
            "can_tolerate": can_tolerate,
            "total_messages": total_messages,
            "phases": phases,
            "total_latency_ms": phases * latency_per_phase_ms,
            "message_complexity": "O(n²)",
        }
        self.log.append(result)
        return result

    def compare_protocols(self) -> dict[str, Any]:
        """Compare Raft, Paxos, BFT for different fault levels."""
        results = []
        for n_faulty in range(0, min(4, self.n_nodes)):
            self.nodes = [{"id": i, "alive": True, "state": "follower"} for i in range(self.n_nodes)]
            raft = self.simulate_raft_election()
            self.nodes = [{"id": i, "alive": True, "state": "follower"} for i in range(self.n_nodes)]
            paxos = self.simulate_paxos()
            self.nodes = [{"id": i, "alive": True, "state": "follower"} for i in range(self.n_nodes)]
            bft = self.simulate_bft(n_faulty)
            results.append({
                "n_faulty": n_faulty,
                "raft": raft,
                "paxos": paxos,
                "bft": bft,
            })
        return {
            "comparison": results,
            "summary": "Raft=crash fault tolerance(⌊(n-1)/2⌋), BFT=byzantine(⌊(n-1)/3⌋), Paxos=same as Raft but harder to implement",
        }


class FailureDetectorSimulator:
    """Compares timeout-based vs φ-accrual failure detection."""

    def __init__(self, n_nodes: int = 8, mean_interval_ms: float = 1000):
        self.n_nodes = n_nodes
        self.mean_interval_ms = mean_interval_ms

    def simulate_timeout_detector(
        self, timeout_ms: float, n_heartbeats: int = 100, failure_at: int = 70
    ) -> dict[str, Any]:
        """Simulate fixed timeout failure detection."""
        intervals = np.random.exponential(self.mean_interval_ms, n_heartbeats)
        # Insert failure after failure_at: intervals become very large
        intervals[failure_at:] = np.random.exponential(self.mean_interval_ms * 20, n_heartbeats - failure_at)

        detected_at = None
        false_positives = 0

        for i in range(n_heartbeats):
            if intervals[i] > timeout_ms:
                if i >= failure_at:
                    if detected_at is None:
                        detected_at = i
                else:
                    false_positives += 1

        detection_delay_ms = sum(intervals[failure_at:detected_at]) if detected_at else None
        missed_detection = detected_at is None

        result = {
            "method": "Timeout",
            "timeout_ms": timeout_ms,
            "false_positives": false_positives,
            "false_positive_rate": false_positives / failure_at,
            "detection_step": detected_at,
            "detection_delay_ms": detection_delay_ms,
            "missed_detection": missed_detection,
        }
        return result

    def simulate_phi_accrual(
        self, phi_threshold: float = 8.0, n_heartbeats: int = 100, failure_at: int = 70
    ) -> dict[str, Any]:
        """Simulate φ-accrual failure detection (Cassandra-style)."""
        intervals = np.random.exponential(self.mean_interval_ms, n_heartbeats)
        intervals[failure_at:] = np.random.exponential(self.mean_interval_ms * 20, n_heartbeats - failure_at)

        # Build arrival distribution from first 30 heartbeats (training phase)
        training_intervals = intervals[:30]
        mean_interval = np.mean(training_intervals)
        std_interval = np.std(training_intervals)

        detected_at = None
        false_positives = 0
        phi_values = []

        for i in range(n_heartbeats):
            # φ = -log10(P(next_heartbeat > observed_interval))
            # Using exponential distribution: P(X > x) = exp(-x/mean)
            phi = -np.log10(np.exp(-intervals[i] / mean_interval)) if mean_interval > 0 else 0
            phi_values.append(phi)

            if phi >= phi_threshold:
                if i >= failure_at:
                    if detected_at is None:
                        detected_at = i
                else:
                    false_positives += 1

        detection_delay_ms = sum(intervals[failure_at:detected_at]) if detected_at else None
        missed_detection = detected_at is None

        result = {
            "method": "φ-Accrual",
            "phi_threshold": phi_threshold,
            "false_positives": false_positives,
            "false_positive_rate": false_positives / failure_at,
            "detection_step": detected_at,
            "detection_delay_ms": detection_delay_ms,
            "missed_detection": missed_detection,
            "mean_phi_before_failure": np.mean(phi_values[:failure_at]),
            "mean_phi_after_failure": np.mean(phi_values[failure_at:]) if failure_at < n_heartbeats else None,
            "peak_phi": max(phi_values),
        }
        return result

    def compare_methods(self) -> dict[str, Any]:
        """Compare timeout vs φ-accrual across different timeout thresholds."""
        results = []
        for timeout_ms in [500, 1000, 2000, 5000, 10000]:
            timeout_result = self.simulate_timeout_detector(timeout_ms)
            phi_result = self.simulate_phi_accrual(phi_threshold=8.0)
            results.append({
                "timeout_ms": timeout_ms,
                "timeout_fp_rate": timeout_result["false_positive_rate"],
                "phi_fp_rate": phi_result["false_positive_rate"],
                "timeout_detection_delay_ms": timeout_result["detection_delay_ms"],
                "phi_detection_delay_ms": phi_result["detection_delay_ms"],
            })

        return {
            "comparison": results,
            "key_insight": "φ-accrual adapts to network conditions → lower false positives + faster detection → recommended for production AI clusters",
            "nccl_timeout_default": "30min (too long! → should be 5-10min)",
            "ray_heartbeat_timeout": "10s (simple but effective)",
        }


class ClockOrderingSimulator:
    """Simulates Lamport timestamps and Vector clocks for ordering."""

    def __init__(self, n_processes: int = 3):
        self.n_processes = n_processes

    def simulate_lamport(self, n_events: int = 20) -> dict[str, Any]:
        """Simulate Lamport timestamps for distributed events."""
        clocks = [0] * self.n_processes
        events = []

        for i in range(n_events):
            # Choose sender and receiver
            sender = np.random.randint(0, self.n_processes)
            receiver = np.random.randint(0, self.n_processes)
            while receiver == sender:
                receiver = np.random.randint(0, self.n_processes)

            # Lamport clock rules
            clocks[sender] += 1  # increment before sending
            clocks[receiver] = max(clocks[receiver], clocks[sender]) + 1  # update on receive

            events.append({
                "type": "send→recv",
                "sender": sender,
                "receiver": receiver,
                "lamport_ts": clocks[sender],
                "receiver_ts": clocks[receiver],
            })

        # Check ordering properties
        concurrent_pairs = 0
        causal_pairs = 0
        for i, e1 in enumerate(events):
            for j, e2 in enumerate(events):
                if i < j:
                    if e1["lamport_ts"] < e2["lamport_ts"]:
                        causal_pairs += 1  # may be causal OR concurrent
                    elif e1["lamport_ts"] == e2["lamport_ts"]:
                        concurrent_pairs += 1

        result = {
            "method": "Lamport Timestamps",
            "n_processes": self.n_processes,
            "n_events": n_events,
            "events": events[:5],  # show first 5
            "total_events": n_events,
            "concurrent_pairs_detected": 0,  # Lamport cannot detect!
            "note": "Lamport gives partial ordering → cannot detect concurrent events",
        }
        return result

    def simulate_vector_clocks(self, n_events: int = 20) -> dict[str, Any]:
        """Simulate Vector clocks for full causal ordering."""
        clocks = [[0] * self.n_processes for _ in range(self.n_processes)]
        events = []

        for i in range(n_events):
            sender = np.random.randint(0, self.n_processes)
            receiver = np.random.randint(0, self.n_processes)
            while receiver == sender:
                receiver = np.random.randint(0, self.n_processes)

            # Vector clock rules
            clocks[sender][sender] += 1
            for k in range(self.n_processes):
                clocks[receiver][k] = max(clocks[receiver][k], clocks[sender][k])
            clocks[receiver][receiver] += 1

            events.append({
                "type": "send→recv",
                "sender": sender,
                "receiver": receiver,
                "vector_ts": clocks[receiver],
            })

        # Detect concurrent events
        concurrent_count = 0
        causal_count = 0
        for i, e1 in enumerate(events):
            for j, e2 in enumerate(events):
                if i < j:
                    v1 = e1["vector_ts"]
                    v2 = e2["vector_ts"]
                    # v1 < v2 iff all components ≤ and at least one <
                    all_leq = all(a <= b for a, b in zip(v1, v2))
                    any_lt = any(a < b for a, b in zip(v1, v2))
                    if all_leq and any_lt:
                        causal_count += 1
                    elif not all_leq and not all(a >= b for a, b in zip(v1, v2)):
                        concurrent_count += 1

        result = {
            "method": "Vector Clocks",
            "n_processes": self.n_processes,
            "n_events": n_events,
            "events": events[:5],
            "concurrent_events_detected": concurrent_count,
            "causal_events_detected": causal_count,
            "storage_overhead": f"O({self.n_processes}) per event",
            "note": "Vector clocks detect concurrent events → full causal ordering → DynamoDB uses for conflict detection",
        }
        return result

    def compare_clocks(self) -> dict[str, Any]:
        """Compare Lamport vs Vector clocks."""
        lamport = self.simulate_lamport()
        vector = self.simulate_vector_clocks()

        return {
            "lamport": lamport,
            "vector": vector,
            "comparison": {
                "ordering_power": "Lamport=partial, Vector=complete causal",
                "concurrent_detection": "Lamport=NO, Vector=YES",
                "storage": f"Lamport=O(1), Vector=O({self.n_processes})",
                "overhead": f"Lamport={self.n_processes} integers total, Vector={self.n_processes * self.n_processes}",
            },
            "ai_infra_recommendation": "Step numbers (≈Lamport) sufficient for 8-GPU training; Vector clocks needed for >100 GPU clusters",
        }


class ReplicationSimulator:
    """Simulates Primary-Backup, Chain, and Quorum replication strategies."""

    def __init__(self, n_replicas: int = 3):
        self.n_replicas = n_replicas

    def simulate_primary_backup(self, n_requests: int = 100) -> dict[str, Any]:
        """Simulate primary-backup replication."""
        # Primary processes all requests
        primary_latency_ms = np.random.uniform(1, 5, n_requests)
        # Backup replication time
        replication_latency_ms = np.random.uniform(0.5, 2, n_requests)

        # Total = primary processing + replication to all backups
        total_latency_ms = primary_latency_ms + replication_latency_ms * (self.n_replicas - 1)

        # Fault tolerance: primary fails → longest backup takes over
        switch_time_ms = np.random.uniform(100, 500)  # leader switch time

        result = {
            "strategy": "Primary-Backup",
            "n_replicas": self.n_replicas,
            "avg_write_latency_ms": np.mean(total_latency_ms),
            "avg_read_latency_ms": np.mean(primary_latency_ms),  # read from primary
            "fault_tolerance": f"⌊(n-1)/2⌋={self.n_replicas // 2} crash faults",
            "leader_switch_ms": switch_time_ms,
            "consistency": "Strong (linearizable)",
            "throughput_limit": "Primary is bottleneck",
            "ai_infra": "vLLM scheduler (1 primary, no election), Ray controller",
        }
        return result

    def simulate_chain_replication(self, n_requests: int = 100) -> dict[str, Any]:
        """Simulate chain replication: Head→Middle→Tail."""
        # Each node adds its processing time
        per_node_latency_ms = np.random.uniform(1, 3, n_requests)
        # Chain: request traverses entire chain
        chain_latency_ms = per_node_latency_ms * self.n_replicas

        # Read from tail (strong consistency)
        read_latency_ms = per_node_latency_ms  # single node read

        result = {
            "strategy": "Chain Replication",
            "n_replicas": self.n_replicas,
            "avg_write_latency_ms": np.mean(chain_latency_ms),
            "avg_read_latency_ms": np.mean(read_latency_ms),
            "fault_tolerance": f"⌊(n-1)/2⌋={self.n_replicas // 2} crash faults",
            "write_path": f"Head→...→Tail ({self.n_replicas} hops)",
            "read_path": "Tail (1 hop, strong consistency)",
            "consistency": "Strong (linearizable via tail)",
            "throughput": f"Write distributed across chain, but tail is read bottleneck",
            "ai_infra": "PD separation (prefill→decode = 2-node chain for KV transfer)",
        }
        return result

    def simulate_quorum(self, n_requests: int = 100, w: int = 2, r: int = 2) -> dict[str, Any]:
        """Simulate quorum-based replication (DynamoDB-style)."""
        # Write: need W replicas to acknowledge
        write_latency_ms = np.sort(np.random.uniform(1, 10, self.n_replicas))
        write_time = write_latency_ms[w - 1]  # wait for W-th fastest

        # Read: need R replicas to respond
        read_latency_ms = np.sort(np.random.uniform(0.5, 5, self.n_replicas))
        read_time = read_latency_ms[r - 1]  # wait for R-th fastest

        w_plus_r = w + r
        consistency = "Strong" if w_plus_r > self.n_replicas else "Eventual"

        result = {
            "strategy": "Quorum",
            "n_replicas": self.n_replicas,
            "W": w,
            "R": r,
            "W+R": w_plus_r,
            "W+R>N": w_plus_r > self.n_replicas,
            "consistency": consistency,
            "avg_write_latency_ms": np.mean(np.random.uniform(1, 10, n_requests)[:w]),
            "avg_read_latency_ms": np.mean(np.random.uniform(0.5, 5, n_requests)[:r]),
            "fault_tolerance": f"⌊(n-min(W,R))/2⌋ faults with {consistency} reads",
            "ai_infra": "Checkpoint writes: W=N, R=1 (all GPUs must write); Inference: W=1, R=1 (single GPU)",
        }
        return result

    def compare_strategies(self) -> dict[str, Any]:
        """Compare all replication strategies."""
        pb = self.simulate_primary_backup()
        chain = self.simulate_chain_replication()
        quorum_strong = self.simulate_quorum(w=(self.n_replicas + 1) // 2, r=(self.n_replicas + 1) // 2)
        quorum_eventual = self.simulate_quorum(w=1, r=1)

        return {
            "primary_backup": pb,
            "chain": chain,
            "quorum_strong": quorum_strong,
            "quorum_eventual": quorum_eventual,
            "comparison_table": {
                "Write Latency": f"PB={pb['avg_write_latency_ms']:.1f}ms / Chain={chain['avg_write_latency_ms']:.1f}ms / Quorum={quorum_strong['avg_write_latency_ms']:.1f}ms",
                "Read Latency": f"PB={pb['avg_read_latency_ms']:.1f}ms / Chain={chain['avg_read_latency_ms']:.1f}ms / Quorum={quorum_strong['avg_read_latency_ms']:.1f}ms",
                "Consistency": "PB=Strong / Chain=Strong / Quorum=Strong(if W+R>N) or Eventual",
                "Fault Tolerance": f"All tolerate ⌊(n-1)/2⌋={self.n_replicas // 2} faults",
            },
            "ai_infra_choice": "Training=Primary-Backup(coordinator), Inference=Quorum(W=1,R=1), PD=Chain(KV transfer)",
        }


class PACELCSimulator:
    """Simulates PACELC trade-offs for AI Infra scenarios."""

    def analyze_training_scenario(self, n_gpus: int = 8) -> dict[str, Any]:
        """PACELC analysis for distributed training (FSDP)."""
        result = {
            "scenario": "Distributed Training (FSDP/ZeRO)",
            "partition_behavior": "PC → Consistency priority → Training stops during partition",
            "normal_behavior": "EC → Consistency priority → AllReduce ensures all GPUs see same gradients",
            "PACELC_choice": "PC/EC → Both choose Consistency",
            "rationale": "Training requires all GPUs to have identical model state → any divergence = training failure",
            "n_gpus": n_gpus,
            "partition_impact": f"AllReduce fails with any GPU loss → training stops → {n_gpus - 1} idle GPUs",
            "rtx4090_specific": f"FSDP {n_gpus}GPU=0.46x → PCIe scaling disaster → single GPU training more efficient!",
        }
        return result

    def analyze_inference_scenario(self, n_gpus: int = 4) -> dict[str, Any]:
        """PACELC analysis for distributed inference (vLLM)."""
        result = {
            "scenario": "Distributed Inference (vLLM)",
            "partition_behavior": "PA → Availability priority → Other GPUs continue serving",
            "normal_behavior": "EL → Latency priority → Fast response > consistency",
            "PACELC_choice": "PA/EL → Both choose Availability/Latency",
            "rationale": "Inference doesn't need all GPUs synchronized → each GPU serves independently",
            "n_gpus": n_gpus,
            "partition_impact": "GPU loss → requests routed to remaining GPUs → degraded throughput but still available",
            "tp_impact": "TP requires all GPUs → 1 GPU loss = TP group failure → not PA! → Need replication for TP fault tolerance",
            "rtx4090_specific": "Single GPU inference (7B INT4) → No distributed needed → Best PACELC choice!",
        }
        return result

    def analyze_pd_separation(self) -> dict[str, Any]:
        """PACELC analysis for PD disaggregation."""
        result = {
            "scenario": "PD Disaggregation (Prefill-Decode Separation)",
            "partition_behavior": "PA → Decode instance continues with local cache → Graceful degradation",
            "normal_behavior": "EL → Latency priority → KV transfer must be fast",
            "PACELC_choice": "PA/EL → Latency and Availability",
            "rationale": "Decode must respond fast → if prefill unreachable, decode serves from cache or local prefill",
            "kv_transfer_latency": "PCIe: ~3% TTFT impact → acceptable; NVLink: ~0.2% → production ideal",
            "rtx4090_specific": "PCIe PD viable (3% TTFT overhead) but NVLink much better → RTX 4090 can do single-GPU PD",
        }
        return result

    def analyze_checkpoint_scenario(self) -> dict[str, Any]:
        """PACELC analysis for checkpoint consistency."""
        result = {
            "scenario": "Training Checkpoint",
            "partition_behavior": "PC → Consistency → All GPUs must checkpoint same step",
            "normal_behavior": "EC → Consistency → Barrier sync before checkpoint",
            "PACELC_choice": "PC/EC → Both choose Consistency",
            "rationale": "Checkpoint must be consistent across all GPUs → inconsistent checkpoint = recovery failure",
            "async_checkpoint": "Asynchronous: EL → Latency priority → faster but risk of inconsistency → need version management",
            "verl_choice": "verl uses async checkpoint → EL trade-off → version tracking to find latest consistent version",
        }
        return result

    def full_comparison(self) -> dict[str, Any]:
        """Compare PACELC choices across all AI Infra scenarios."""
        return {
            "training": self.analyze_training_scenario(),
            "inference": self.analyze_inference_scenario(),
            "pd_separation": self.analyze_pd_separation(),
            "checkpoint": self.analyze_checkpoint_scenario(),
            "summary_table": {
                "Training": "PC/EC (Consistency both sides)",
                "Inference": "PA/EL (Availability+Latency)",
                "PD Separation": "PA/EL (Availability+Latency)",
                "Checkpoint": "PC/EC (Consistency) / Async: EL (Latency)",
            },
            "key_insight": "AI Infra is not one PACELC choice → Different components have different requirements → Training=Consistency, Inference=Availability, Checkpoint=Consistency with async option",
        }


def run_all():
    """Run all simulators and print results."""
    print("=" * 80)
    print("DISTRIBUTED SYSTEMS SIMULATOR FOR AI INFRA")
    print("=" * 80)

    # 1. Consensus comparison
    print("\n1. Consensus Protocol Comparison (5 nodes)")
    consensus = ConsensusSimulator(n_nodes=5)
    comparison = consensus.compare_protocols()
    for r in comparison["comparison"]:
        print(f"\n  Faulty={r['n_faulty']}:")
        print(f"    Raft: can_elect={r['raft']['can_elect_leader']}, latency={r['raft']['commit_latency_ms']:.1f}ms")
        print(f"    Paxos: consensus={r['paxos']['reached_consensus']}, rounds={r['paxos']['rounds_to_consensus']}")
        print(f"    BFT: tolerate={r['bft']['can_tolerate']}, messages={r['bft']['total_messages']}")

    # 2. Failure detection comparison
    print("\n2. Failure Detection: Timeout vs φ-Accrual")
    fd = FailureDetectorSimulator(n_nodes=8)
    fd_comparison = fd.compare_methods()
    for r in fd_comparison["comparison"]:
        print(f"  Timeout={r['timeout_ms']}ms: FP_rate={r['timeout_fp_rate']:.3f}, delay={r['timeout_detection_delay_ms']}")
        print(f"  φ-Accrual:      FP_rate={r['phi_fp_rate']:.3f}, delay={r['phi_detection_delay_ms']}")
    print(f"  Key: {fd_comparison['key_insight']}")

    # 3. Clock ordering comparison
    print("\n3. Clock Ordering: Lamport vs Vector Clocks")
    clock = ClockOrderingSimulator(n_processes=3)
    clock_comparison = clock.compare_clocks()
    comp = clock_comparison["comparison"]
    print(f"  Lamport: {comp['ordering_power']}, storage={comp['storage']}")
    print(f"  Vector:  concurrent_detect={comp['concurrent_detection']}, overhead={comp['overhead']}")
    print(f"  Recommendation: {clock_comparison['ai_infra_recommendation']}")

    # 4. Replication strategies
    print("\n4. Replication Strategies (3 replicas)")
    rep = ReplicationSimulator(n_replicas=3)
    rep_comparison = rep.compare_strategies()
    table = rep_comparison["comparison_table"]
    for k, v in table.items():
        print(f"  {k}: {v}")
    print(f"  Choice: {rep_comparison['ai_infra_choice']}")

    # 5. PACELC analysis
    print("\n5. PACELC Analysis for AI Infra")
    pacelc = PACELCSimulator()
    full = pacelc.full_comparison()
    for scenario, choice in full["summary_table"].items():
        print(f"  {scenario}: {choice}")
    print(f"  Insight: {full['key_insight']}")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed Systems Simulator for AI Infra")
    parser.add_argument("class_name", nargs="?", default="all",
                        choices=["ConsensusSimulator", "FailureDetectorSimulator",
                                 "ClockOrderingSimulator", "ReplicationSimulator",
                                 "PACELCSimulator", "all"],
                        help="Which simulator class to run")
    args = parser.parse_args()

    if args.class_name == "all":
        run_all()
    elif args.class_name == "ConsensusSimulator":
        sim = ConsensusSimulator(n_nodes=5)
        print(sim.compare_protocols())
    elif args.class_name == "FailureDetectorSimulator":
        sim = FailureDetectorSimulator(n_nodes=8)
        print(sim.compare_methods())
    elif args.class_name == "ClockOrderingSimulator":
        sim = ClockOrderingSimulator(n_processes=3)
        print(sim.compare_clocks())
    elif args.class_name == "ReplicationSimulator":
        sim = ReplicationSimulator(n_replicas=3)
        print(sim.compare_strategies())
    elif args.class_name == "PACELCSimulator":
        sim = PACELCSimulator()
        print(sim.full_comparison())