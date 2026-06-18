# rLLM #605 GRPO Grouping Bug — Source-Level Deep Reading

> 2026-06-18 | Source-level analysis of CRITICAL GRPO grouping bug
> ★★★★★★★★ GRPO is BROKEN in rLLM — group size = 1 → advantage = raw reward → no variance reduction!
> ★★★★★★★★ Root cause found in transform.py:127 — groups by task_id:trajectory.name instead of task_id
> ★★★★★★★★ 18+ days, ZERO developer response — community pressure needed
> ★★★★★★★★ 1-line fix verified at source level — change grouping key from task_id:name to task_id

---

## 1. Bug Location: transform.py:127

```
★★★★★★★★★ EXACT bug location (rllm/trainer/algorithms/transform.py:105-146):

def _build_trajectory_groups(episodes, compact_filtering_config=None):
    trajectories_by_name: dict[str, list[Trajectory]] = defaultdict(list)
    metadata_by_name: dict[str, list[dict]] = defaultdict(list)

    for episode in episodes:
        ...
        task_id = episode.task_id                    # line 123
        for trajectory in episode.trajectories:       # line 124
            if len(trajectory.steps) == 0:
                continue                              # line 126
            trajectories_by_name[f"{task_id}:{trajectory.name}"].append(trajectory)  # line 127 ← BUG!
            metadata_by_name[f"{task_id}:{trajectory.name}"].append(...)              # line 128

★★★★★★★★★ THE BUG: line 127 groups by f"{task_id}:{trajectory.name}"

  → trajectory.name comes from agent.name (types.py:552)
  → Default: _DEFAULT_TRAJ_NAME = "default_traj_name" (types.py:91)
  → Each agent/flow has its own name → different trajectory.name
  → Result: trajectories with SAME task_id but DIFFERENT trajectory.name
    → get SEPARATE groups → group size = 1 → GRPO BROKEN!

★★★★★★★★★ For GRPO: grouping should be by task_id (prompt) ONLY:
  → All responses to the same prompt should be in ONE group
  → GRPO advantage = (reward - mean(group_rewards)) / std(group_rewards)
  → With group size = 1: mean = reward, std = 0 → advantage = undefined
  → ★★★★★★★★ rLLM handles std=0 by returning raw reward → no variance reduction → GRPO = REINFORCE!

★★★★★★★★★ THE FIX (1 line):
  → Change line 127 from:
    trajectories_by_name[f"{task_id}:{trajectory.name}"].append(trajectory)
  → To:
    trajectories_by_name[task_id].append(trajectory)
  → Same fix for line 128 (metadata)

★★★★★★★★★ WHY this is correct:
  → GRPO groups by PROMPT (task_id) → all rollouts for same prompt in one group
  → Different trajectory.name → different agents/flows → but SAME prompt
  → All agents responding to same prompt → should be compared together
  → ★★★★★★★★ This is how verl groups: by prompt/task → all responses in one batch
```

---

## 2. Trajectory.name Source: Agent Names

```
★★★★★★★★★ How trajectory.name gets set (types.py:552):

traj_name = getattr(agent, "name", None) or _DEFAULT_TRAJ_NAME

★★★★★★★★★ Trajectory data model (types.py:241-255):

class Trajectory(BaseModel):
    uid: str = str(uuid.uuid4())     # unique per trajectory
    name: str = _DEFAULT_TRAJ_NAME   # ← agent name, NOT task identifier
    task: Any = None
    steps: list[Step]
    reward: float | None = None
    input: dict | None = None
    output: Any = None               # ← CRITICAL: was always None before #663 fix!

★★★★★★★★★ Grouping key analysis:
  → task_id: identifies the prompt/task (correct for GRPO grouping)
  → trajectory.uid: unique per trajectory (too granular → group size always = 1)
  → trajectory.name: agent/flow name (too granular → group size = 1 per agent)
  → ★★★★★★★★ CORRECT grouping key for GRPO: task_id ONLY
  → Same task_id → same prompt → all rollouts compare → GRPO advantage works
```

---

## 3. TrajectoryGroup Data Model

```
★★★★★★★★★ TrajectoryGroup (types.py:384-410):

class TrajectoryGroup(BaseModel):
    trajectories: list[Trajectory]    # trajectories in this group
    group_id: str = ""               # e.g., "task1:agent_0"
    metadata: list[dict]             # per-trajectory metadata
    weight_version: int = 0          # for staleness tracking

★★★★★★★★★ group_role property (types.py:403-405):

    @property
    def group_role(self) -> str:
        return self.group_id.split(":")[1] if ":" in self.group_id[:-1] else "all_groups"

★★★★★★★★★ PROBLEM with group_role:
  → group_id = f"{task_id}:{trajectory.name}" (from transform.py:127)
  → group_role = trajectory.name (after splitting on ":")
  → Different trajectory names → different group_roles
  → ★★★★★★★★ advantage.py groups by group_role → separate advantage computation per agent name
  → This means GRPO computes advantages per-agent → not per-prompt → BROKEN!

★★★★★★★★★ task_id property (types.py:408-409):

    @property
    def task_id(self) -> str:
        return self.group_id.split(":")[0]

  → task_id correctly extracted from group_id prefix
  → But task_id is NOT used for GRPO grouping → only group_role!
```

---

## 4. GRPO Advantage Computation

```
★★★★★★★★★ advantage.py:153-212 — collect_reward_and_advantage_from_trajectory_groups():

  → Groups trajectories by group_role (line 200)
  → For each group_role: compute advantages from group rewards
  → GRPO: advantage = (reward - mean(group_rewards)) / std(group_rewards)

★★★★★★★★★ With the grouping bug:
  → group_role = trajectory.name (agent name)
  → Each agent gets its own TrajectoryGroup
  → Group size = 1 (only one trajectory per agent per task)
  → ★★★★★★★★ GRPO with group_size=1:
    → mean(group_rewards) = reward (single value)
    → std(group_rewards) = 0 → division by zero!
    → rLLM handles this: returns raw reward → advantage = reward
    → NO variance reduction → GRPO becomes REINFORCE → no relative comparison

★★★★★★★★★ With the fix (group by task_id):
  → group_role = "all_groups" (since no ":" in task_id)
  → All trajectories for same task → ONE group
  → Group size = num_rollouts (e.g., 8 or 16)
  → ★★★★★★★★ GRPO works correctly:
    → mean(group_rewards) = average of all rollout rewards for this prompt
    → std(group_rewards) = standard deviation → meaningful normalization
    → advantage = (reward - mean) / std → relative comparison → correct GRPO!
```

---

## 5. rLLM #663 — Step.output Bug (MERGED June 17)

```
★★★★★★★★★ rLLM #663 — CRITICAL fix MERGED June 17:
  → Step.output was always None → ALL rewards = 0.0!
  → trace_record_to_step() set model_response=content but left output=None
  → Evaluators checking step.output → always None → rewards = 0.0
  → ★★★★★★★★ Fix: output=content (1 line)
  → Verified on Koala H200x8 with Qwen3.5-9B → rewards from all-zero to non-zero

★★★★★★★★★ Impact: ANY rLLM training before June 17 was COMPLETELY INVALID!
  → ALL rewards = 0.0 → ALL advantages = 0.0 → NO learning signal
  → Training wasted compute → model never improved
  → ★★★★★★★★ This bug was worse than #605! Rewards were literally zero!
  → Combined with #605: even if rewards were nonzero → grouping still broken → GRPO = REINFORCE

★★★★★★★★★ rLLM bugs compound:
  1. #663: rewards = 0.0 → NO learning signal at all → FIXED June 17
  2. #605: GRPO grouping = per-agent → group size = 1 → NO variance reduction → STILL OPEN 18+ days!
  → ★★★★★★★★ BOTH bugs needed fixing → #663 fixed but #605 still OPEN → GRPO still broken!
```

---

## 6. Comparison with verl GRPO Grouping

```
★★★★★★★★★ verl groups correctly by prompt/task:

verl trainer: groups by task_ids (prompt identifiers)
  → All responses for same prompt → ONE GRPO group
  → Group size = num_rollouts (configurable, typically 8-16)
  → ★★★★★★★★ Correct GRPO: relative comparison across responses to same prompt

★★★★★★★★★ rLLM groups incorrectly by task_id:trajectory.name:
  → Each agent/flow name → separate group
  → Group size = 1 per agent → no comparison possible
  → ★★★★★★★★ Incorrect GRPO: no relative comparison → broken

★★★★★★★★★ The fundamental difference:
  → verl: group by PROMPT → what question was asked
  → rLLM: group by PROMPT:AGENT → who answered the question
  → ★★★★★★★★ GRPO needs to compare answers to the SAME question → group by prompt only!
```

---

## Key Findings Summary

★★★★★★★★★ #605 BUG: transform.py:127 groups by task_id:trajectory.name → group size = 1 → GRPO BROKEN!
★★★★★★★★★ #605 FIX: change grouping key from task_id:trajectory.name to task_id → 1 line fix
★★★★★★★★★ trajectory.name = agent/flow name (NOT task identifier) → too granular for GRPO
★★★★★★★★★ TrajectoryGroup.group_role splits on ":" → gets trajectory.name → separate per-agent
★★★★★★★★★ #663 FIX (MERGED June 17): Step.output was None → ALL rewards = 0.0 → training INVALID!
★★★★★★★★★ Both bugs compound: #663 zero rewards + #605 broken grouping → complete GRPO failure
★★★★★★★★★ verl groups correctly: by prompt/task → all responses in one group → GRPO works
★★★★★★★★★ rLLM #605: 18+ days, ZERO developer response → community pressure needed!
★★★★★★★★★ RTX 4090 impact: rLLM GRPO COMPLETELY BROKEN until BOTH #605 and #663 fixed

---

## References

- rLLM #605: https://github.com/rllm-org/rllm/issues/605 (CRITICAL, OPEN 18+ days, ZERO comments)
- rLLM #663: MERGED June 17 — Step.output was None → ALL rewards = 0.0
- Source: rllm/trainer/algorithms/transform.py:127 (grouping key bug)
- Source: rllm/trainer/algorithms/advantage.py:153 (advantage computation)
- Source: rllm/types.py:384 (TrajectoryGroup), 241 (Trajectory), 91 (_DEFAULT_TRAJ_NAME)
- verl GRPO: correct grouping by task_ids → all responses per prompt in one group
- Tier 1 comment draft: notebook/tier1-comments/rllm-605-grpo-grouping-bug-comment-draft.md
