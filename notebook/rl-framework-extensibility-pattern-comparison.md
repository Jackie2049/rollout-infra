# RL Framework Extensibility Pattern Comparison: verl × rLLM × TRL

**Date**: 2026-07-15 (Session 10)
**Purpose**: Compare how 3 GRPO frameworks handle extensibility (adding new estimators, losses, trainers)
**Sources**: rLLM terminal-rl plugin system, verl V1 registry, TRL Trainer inheritance

---

## 1. Pattern Taxonomy

| Framework | Registration Mechanism | Scope | Discovery | Override |
|-----------|----------------------|-------|-----------|----------|
| **verl V1** | `@register_trainer(name)` decorator → `TRAINER_REGISTRY` dict | Trainer types only (sync/colocate_async/separate_async) | `get_trainer_cls(name)` → instantiate by string | Config YAML: `trainer_type: "sync"` |
| **rLLM terminal-rl** | `@rllm.register_*` family of decorators → module-level dicts | Estimators + losses + trainers + runners + models + reward functions | `rllm.get_*` family → instantiate by string | Config YAML: `adv_estimator: "grpo"` |
| **TRL** | Class inheritance (`GRPOTrainer extends Trainer`) | Trainer class only (no estimator/loss registry) | Direct import `from trl import GRPOTrainer` | Config dict: `loss_type: "ppo_clip"` |

---

## 2. verl V1 Registry Pattern

### Code Structure
```python
# verl/trainer/ppo/v1/trainer_base.py

TRAINER_REGISTRY = {}  # Module-level dict

def register_trainer(name: str):
    """Decorator to register trainer classes."""
    def decorator(cls):
        TRAINER_REGISTRY[name] = cls
        return cls
    return decorator

def get_trainer_cls(name: str):
    """Look up trainer by name."""
    if name not in TRAINER_REGISTRY:
        raise ValueError(f"Trainer {name} not found. Available: {list(TRAINER_REGISTRY.keys())}")
    return TRAINER_REGISTRY[name]

# Registration
@register_trainer("sync")
class SyncPPOTrainer(PPOTrainerBase):
    ...

@register_trainer("colocate_async")
class ColocateAsyncPPOTrainer(PPOTrainerBase):
    ...

@register_trainer("separate_async")
class SeparateAsyncPPOTrainer(PPOTrainerBase):
    ...
```

### Key Properties
- **3 trainer types** registered: sync, colocate_async, separate_async
- **Estimators and losses** are NOT registered — they're hardcoded methods on PPOTrainerBase
- **Advantage computation**: 14 estimators via `compute_*_advantage` method family (not decorator-based)
- **Loss computation**: `core_algos.py` imports → direct function calls
- **Adding UP-GRPO**: Had to modify `core_algos.py` + `trainer_base.py` → no clean extension point

### Limitation
verl's registry only covers trainer types. To add a new advantage estimator or loss, you must:
1. Add the computation function to `core_algos.py`
2. Add the dispatch logic in `trainer_base.py._compute_advantage()`
3. Add the dispatch logic in `trainer_base.py._compute_policy_loss()`

This is invasive — no plugin isolation. Our UP-GRPO PR (#9) required modifying 3 files.

---

## 3. rLLM terminal-rl Plugin Architecture

### Code Structure
```python
# rllm/registry.py

_REGISTRY = {
    "adv_estimator": {},
    "loss": {},
    "trainer": {},
    "runner": {},
    "model": {},
    "reward_fn": {},
}

def register_adv_estimator(name: str):
    def decorator(cls):
        _REGISTRY["adv_estimator"][name] = cls
        return cls
    return decorator

def register_loss(name: str):
    def decorator(cls):
        _REGISTRY["loss"][name] = cls
        return cls
    return decorator

# ... similar for trainer, runner, model, reward_fn

# Usage
@rllm.register_adv_estimator("grpo")
class GRPOAdvEstimator(AdvEstimatorBase):
    def compute(self, rewards, **kwargs):
        ...

@rllm.register_adv_estimator("rloo")
class RLOOAdvEstimator(AdvEstimatorBase):
    def compute(self, rewards, **kwargs):
        ...

@rllm.register_loss("ppo_clip")
class PPOClipLoss(LossBase):
    def compute(self, ratio, advantage, **kwargs):
        ...
```

### Key Properties
- **6 registration categories**: adv_estimator, loss, trainer, runner, model, reward_fn
- **6 advantage estimators**: grpo, reinforce, reinforce_bl, rloo, prpo, echo
- **9 policy losses**: ppo_clip, dppo_tv, dppo_kl, cispo, gspo, icepop, reinforce, reinforce_kl, echo
- **LossContext**: cross-backend context object that carries advantage estimates → loss receives context
- **Native-first routing**: estimator produces context → loss reads from context → no tight coupling

### Extensibility
Adding a new estimator or loss requires:
1. Create a new class inheriting from the appropriate base
2. Add `@rllm.register_*("name")` decorator
3. Reference by name in config YAML

**Zero modification to existing code** → clean plugin isolation. This is the most extensible design.

### Our PR #2 Impact
Our GROUPED_GRPO PR adds `grouping_key` parameter to the existing `@rllm.register_adv_estimator("grpo")` class. The plugin pattern means we only modify one class — no cascading changes needed.

---

## 4. TRL Inheritance Pattern

### Code Structure
```python
# trl/grpo_trainer.py

class GRPOTrainer(Trainer):
    """GRPO trainer via inheritance from HuggingFace Trainer."""

    def __init__(self, ...):
        self.loss_type = loss_type  # "ppo_clip" or other

    def _compute_advantage(self, rewards):
        """Hardcoded GRPO advantage computation."""
        # Mean-subtraction + std-normalization
        # No plugin, no registry — just a method

    def _compute_loss(self, ...):
        """Dispatch based on loss_type string."""
        if self.loss_type == "ppo_clip":
            return self._ppo_clip_loss(...)
        # No other losses available in base GRPOTrainer!
```

### Key Properties
- **Single trainer class**: GRPOTrainer inherits from HF Trainer
- **No estimator registry**: advantage computation is a hardcoded method
- **No loss registry**: only PPO-clip available (loss_type dispatch is minimal)
- **Adding new estimators/losses**: subclass GRPOTrainer and override methods

### Limitation
TRL's design is the least extensible:
- To add a new advantage estimator → override `_compute_advantage` in a subclass
- To add a new loss → override `_compute_loss` in a subclass
- No composition → inheritance creates deep class hierarchies
- HF Trainer is 3000+ lines → subclassing is fragile (many implicit dependencies)

---

## 5. Pattern Comparison: Adding UP-GRPO

| Step | verl V1 | rLLM terminal-rl | TRL |
|------|---------|-------------------|-----|
| **New estimator needed?** | No (use GRPO) | No (use GRPO) | No (use GRPO) |
| **New loss needed?** | Yes | Yes | Yes |
| **Where to add?** | Modify `core_algos.py` + `trainer_base.py` | Add `@rllm.register_loss("up_grpo")` class | Override `_compute_loss` in subclass |
| **Files modified** | 3 | 1 (new class) | 1 (new subclass) |
| **Existing code changes** | Yes (dispatch logic) | No (pure addition) | No (override) |
| **Config access** | YAML: `loss_type: "up_grpo"` | YAML: `loss: "up_grpo"` | Dict: `loss_type: "up_grpo"` |
| **Test isolation** | Moderate | Excellent (independent class) | Poor (subclass dependencies) |

**Key insight**: rLLM's plugin pattern is the best for extensibility — adding UP-GRPO requires zero changes to existing code, only a new class file. verl requires modifying existing dispatch logic. TRL requires subclassing.

---

## 6. Architecture Comparison

```
verl V1:
  PPOTrainerBase (71KB base class)
    ├── @register_trainer("sync") → SyncPPOTrainer
    ├── @register_trainer("colocate_async") → ColocateAsyncPPOTrainer
    ├── @register_trainer("separate_async") → SeparateAsyncPPOTrainer
    ├── Advantage: 14 methods on PPOTrainerBase (not registered)
    ├── Loss: functions in core_algos.py (not registered)
    └── Config: YAML selects trainer type, estimator/loss hardcoded

rLLM terminal-rl:
  AdvEstimatorBase ← @register_adv_estimator
    ├── "grpo" → GRPOAdvEstimator
    ├── "rloo" → RLOOAdvEstimator
    ├── "reinforce" → REINFORCEAdvEstimator
    ├── "echo" → ECHOAdvEstimator (uses GRPO internally)
    └── LossContext connects estimator → loss
  LossBase ← @register_loss
    ├── "ppo_clip" → PPOClipLoss
    ├── "cispo" → CISPOLoss
    ├── "up_grpo" → (our potential contribution)
    └── All receive LossContext from estimator
  Config: YAML selects estimator + loss independently

TRL:
  Trainer (3000+ lines HF base)
    └── GRPOTrainer ← inheritance
          ├── _compute_advantage (hardcoded GRPO)
          ├── _compute_loss (dispatch by string, limited)
          └── loss_type = "ppo_clip" (only option in base)
  Adding UP-GRPO: subclass GRPOTrainer → override _compute_loss
```

---

## 7. Design Pattern Lessons

### Lesson 1: Registry > Inheritance for Extensibility
- Registry pattern allows zero-modification additions
- Inheritance pattern requires subclassing + understanding base class internals
- verl hybrid: registry for trainers, inheritance for estimator/loss dispatch → inconsistent

### Lesson 2: Context Objects Enable Composition
- rLLM's `LossContext` carries advantage estimates → loss reads from context
- This decouples estimator from loss → they can be combined independently
- verl: advantage is a tensor passed directly → no composition flexibility

### Lesson 3: Config-Driven Discovery Enables Experimentation
- rLLM: `adv_estimator: "grpo"`, `loss: "cispo"` → any combination possible
- verl: estimator selected by function name, loss by config flag → limited combinations
- TRL: estimator hardcoded, loss by string → few combinations

### Lesson 4: Module-Level Dict Registration is Pythonic
- All 3 frameworks use `dict` as registry (not classes, not metaclasses)
- Decorator pattern: `@register_X(name)` → standard Python pattern
- Lazy discovery: registry populated at import time, queried at runtime
- This is the standard pattern in ML frameworks (PyTorch `torch.nn.Module`, HF `AutoModel`)

---

## 8. RTX 4090 Implications

For RTX 4090 GRPO training, extensibility matters because:

1. **UP-GRPO loss** (our PR #9) requires adding a new loss → verl required 3-file modification, rLLM would need only 1-file addition
2. **Custom advantage estimators** for shaped rewards → rLLM's plugin pattern allows easy experimentation
3. **Loss-estimator combinations** → rLLM allows any combination, verl/TRL limit combinations
4. **Future: CISPO, GSPO** → rLLM already has these registered, verl would need manual additions

**Recommendation**: For rapid experimentation on RTX 4090, rLLM's plugin pattern is superior. For production deployment, verl's more rigid architecture provides stability.

---

## Session Stats
- **3 frameworks** extensibility patterns compared (verl, rLLM, TRL)
- **Registry vs Inheritance** pattern analysis with UP-GRPO as case study
- **4 design lessons** extracted (registry > inheritance, context objects, config-driven, module-level dict)
- **RTX 4090 recommendation**: rLLM for experimentation, verl for production
