#!/usr/bin/env python3
"""
AI Ethics & Governance Simulator — Fairness, Privacy, Transparency Assessment

Implements the core concepts from ai-ethics-governance-deep-dive.md:
1. Fairness assessment → 4 definitions (DemographicParity/EqualizedOdds/Individual/Calibration)
2. Privacy modeling → DP-SGD ε-privacy calculation → privacy-utility tradeoff
3. Transparency scoring → model interpretability + documentation assessment
4. Governance compliance → EU AI Act risk classification + ASL risk levels

No GPU required — pure CPU-based ethics & governance simulation.
Key insight: AI Infra engineer = governance executor → technical implementation of ethical principles!
"""

import json
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple


# ============================================================================
# Part 1: Fairness Assessment — 4 Definitions
# ============================================================================

class FairnessAssessment:
    """Assess model fairness across 4 mathematical definitions.

    Key insight from ethics deep dive:
    → 4 fairness definitions → each captures different notion → CONFLICT!
    → → DemographicParity: P(ŷ=1|A=0) = P(ŷ=1|A=1) → equal outcome rates
    → → EqualizedOdds: P(ŷ=1|Y=1,A=0) = P(ŷ=1|Y=1,A=1) → equal error rates
    → → IndividualFairness: similar individuals → similar predictions → counterfactual
    → → Calibration: P(Y=1|ŷ=0.9,A=0) = P(Y=1|ŷ=0.9,A=1) → equal confidence accuracy

    → → → Impossible to satisfy all simultaneously! (Chouldechova impossibility theorem)
    → → → → Must choose which fairness definition based on context → no universal answer!
    """

    # Simulated model predictions for different demographic groups
    # Format: (true_label, predicted_prob, group_id)
    DEMOGRAPHIC_GROUPS = {
        "group_A": {"size": 1000, "base_rate": 0.30, "model_accuracy": 0.85},
        "group_B": {"size": 800, "base_rate": 0.20, "model_accuracy": 0.78},
        "group_C": {"size": 600, "base_rate": 0.40, "model_accuracy": 0.82},
    }

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def generate_predictions(self) -> Dict[str, List[Tuple]]:
        """Generate simulated predictions for each demographic group."""
        predictions = {}
        for group, config in self.DEMOGRAPHIC_GROUPS.items():
            preds = []
            for i in range(config["size"]):
                true_label = 1 if random.random() < config["base_rate"] else 0
                # Model predictions influenced by accuracy
                if true_label == 1:
                    predicted_prob = random.gauss(0.7, 0.15)
                else:
                    predicted_prob = random.gauss(0.3, 0.15)
                predicted_prob = max(0, min(1, predicted_prob))
                preds.append((true_label, predicted_prob, group))
            predictions[group] = preds
        return predictions

    def compute_demographic_parity(self, predictions: Dict,
                                    threshold: float = 0.5) -> Dict:
        """P(ŷ=1|A=group) should be equal across groups."""
        rates = {}
        for group, preds in predictions.items():
            positive_rate = sum(1 for _, p, _ in preds if p >= threshold) / len(preds)
            rates[group] = positive_rate

        # Fairness gap = max difference between groups
        max_rate = max(rates.values())
        min_rate = min(rates.values())
        gap = max_rate - min_rate

        return {
            "definition": "DemographicParity",
            "description": "P(ŷ=1|A=group) equal across groups → equal outcome rates",
            "positive_rates": rates,
            "gap": gap,
            "fair": gap < 0.05,  # 5% tolerance
        }

    def compute_equalized_odds(self, predictions: Dict,
                                threshold: float = 0.5) -> Dict:
        """P(ŷ=1|Y=y,A=group) equal across groups for y=0 and y=1."""
        tpr_rates = {}  # True Positive Rate: P(ŷ=1|Y=1,A)
        fpr_rates = {}  # False Positive Rate: P(ŷ=1|Y=0,A)

        for group, preds in predictions.items():
            positives = [(t, p) for t, p, g in preds if t == 1]
            negatives = [(t, p) for t, p, g in preds if t == 0]

            tpr = sum(1 for _, p in positives if p >= threshold) / max(1, len(positives))
            fpr = sum(1 for _, p in negatives if p >= threshold) / max(1, len(negatives))

            tpr_rates[group] = tpr
            fpr_rates[group] = fpr

        tpr_gap = max(tpr_rates.values()) - min(tpr_rates.values())
        fpr_gap = max(fpr_rates.values()) - min(fpr_rates.values())

        return {
            "definition": "EqualizedOdds",
            "description": "P(ŷ=1|Y=y,A) equal for y=0,1 → equal error rates",
            "tpr_rates": tpr_rates,
            "fpr_rates": fpr_rates,
            "tpr_gap": tpr_gap,
            "fpr_gap": fpr_gap,
            "fair": tpr_gap < 0.05 and fpr_gap < 0.05,
        }

    def compute_calibration(self, predictions: Dict,
                            bins: int = 10) -> Dict:
        """P(Y=1|ŷ=p,A) should equal p for all groups → confidence = accuracy."""
        calibration = {}
        for group, preds in predictions.items():
            bin_stats = []
            for b in range(bins):
                low = b / bins
                high = (b + 1) / bins
                in_bin = [(t, p) for t, p, g in preds if low <= p < high]
                if len(in_bin) > 0:
                    avg_pred = sum(p for _, p in in_bin) / len(in_bin)
                    actual_rate = sum(1 for t, _ in in_bin if t == 1) / len(in_bin)
                    bin_stats.append({
                        "bin": f"{low:.1f}-{high:.1f}",
                        "avg_predicted": avg_pred,
                        "actual_positive_rate": actual_rate,
                        "calibration_error": abs(avg_pred - actual_rate),
                        "count": len(in_bin),
                    })
            avg_error = sum(b["calibration_error"] for b in bin_stats) / len(bin_stats) if bin_stats else 0
            calibration[group] = {"bins": bin_stats, "avg_calibration_error": avg_error}

        # Compare calibration errors across groups
        errors = {g: c["avg_calibration_error"] for g, c in calibration.items()}
        gap = max(errors.values()) - min(errors.values())

        return {
            "definition": "Calibration",
            "description": "P(Y=1|ŷ=p,A) = p → confidence matches accuracy",
            "per_group": calibration,
            "errors": errors,
            "gap": gap,
            "fair": gap < 0.03,  # 3% tolerance
        }

    def compute_individual_fairness(self, predictions: Dict) -> Dict:
        """Similar individuals should get similar predictions → Lipschitz condition."""
        # Simplified: check if prediction variance within groups is reasonable
        # (True individual fairness requires counterfactual pairs → complex)
        variance = {}
        for group, preds in predictions.items():
            pred_probs = [p for _, p, _ in preds]
            mean_prob = sum(pred_probs) / len(pred_probs)
            var = sum((p - mean_prob) ** 2 for p in pred_probs) / len(pred_probs)
            variance[group] = var

        gap = max(variance.values()) - min(variance.values())

        return {
            "definition": "IndividualFairness",
            "description": "Similar individuals → similar predictions (Lipschitz)",
            "variances": variance,
            "gap": gap,
            "fair": gap < 0.02,  # tolerance
            "note": "True individual fairness requires counterfactual pairs → simplified here",
        }

    def full_assessment(self) -> Dict:
        """Run all 4 fairness assessments."""
        predictions = self.generate_predictions()

        dp = self.compute_demographic_parity(predictions)
        eo = self.compute_equalized_odds(predictions)
        cal = self.compute_calibration(predictions)
        ind = self.compute_individual_fairness(predictions)

        return {
            "demographic_parity": dp,
            "equalized_odds": eo,
            "calibration": cal,
            "individual_fairness": ind,
            "impossibility_note": "Cannot satisfy all definitions simultaneously "
                                 "(Chouldechova impossibility theorem → must choose context-appropriate definition)",
        }


# ============================================================================
# Part 2: Privacy Modeling — DP-SGD ε-Privacy
# ============================================================================

class PrivacyModel:
    """Model differential privacy (DP-SGD) privacy-utility tradeoff.

    Key insight from ethics deep dive:
    → DP-SGD: add noise to gradients → ε-privacy guarantee → mathematical!
    → → ε: privacy budget → smaller ε = more private = less utility!
    → → → ε=1: strong privacy → utility drops ~5-10%
    → → → → ε=10: moderate privacy → utility drops ~1-3%
    → → → → → ε=∞: no privacy → full utility → but no guarantee!

    → Composition theorem: total ε = sum of per-step ε → more steps = more budget consumed!
    → → → 1000 steps × ε_per_step = total ε → need to manage budget!
    → → → → Privacy accountant: track total ε across training → stop when budget exhausted!

    → RTX 4090 impact: DP-SGD noise addition = per-gradient op → GPU kernel → ~5% overhead
    → → → DP-SGD on GPU = practical → overhead small → privacy-utility tradeoff is real decision!
    """

    def compute_epsilon(self, noise_multiplier: float, num_steps: int,
                        delta: float = 1e-5, sampling_rate: float = 0.01) -> Dict:
        """Compute ε using simplified DP-SGD accounting (moments accountant)."""
        # Simplified: ε ≈ noise_multiplier × sqrt(num_steps × sampling_rate)
        # Real: use RDP (Rényi Differential Privacy) accountant → more precise
        # For demonstration: use simplified Gaussian mechanism composition

        # Per-step ε (Gaussian mechanism)
        per_step_epsilon = math.sqrt(2 * math.log(1.25 / delta)) / noise_multiplier

        # Total ε (basic composition: ε_total = num_steps × per_step × sampling_rate)
        # Using moments accountant (better bound):
        # ε_total ≈ noise_multiplier × sqrt(2 × num_steps × sampling_rate × log(1/delta))
        total_epsilon = noise_multiplier * math.sqrt(
            2 * num_steps * sampling_rate * math.log(1 / delta)
        )

        # More accurate: RDP-based computation
        rdp_epsilon = self._compute_rdp_epsilon(noise_multiplier, num_steps, sampling_rate, delta)

        return {
            "noise_multiplier": noise_multiplier,
            "num_steps": num_steps,
            "sampling_rate": sampling_rate,
            "delta": delta,
            "per_step_epsilon": per_step_epsilon,
            "basic_composition_epsilon": total_epsilon,
            "rdp_epsilon": rdp_epsilon,
            "privacy_level": self._classify_privacy(rdp_epsilon),
        }

    def _compute_rdp_epsilon(self, sigma: float, steps: int, q: float,
                              delta: float) -> float:
        """Simplified RDP-based epsilon computation."""
        # RDP: α-order Rényi divergence → convert to (ε,δ)-DP
        # For Gaussian mechanism: RDP_α = q²α/(2σ²) per step
        # Compose: RDP_α_total = steps × RDP_α_per_step
        # Convert to (ε,δ)-DP: ε = RDP_α + log(1/δ)/(α-1)

        best_epsilon = float('inf')
        for alpha in [2, 4, 8, 16, 32, 64, 128, 256]:
            rdp_per_step = q ** 2 * alpha / (2 * sigma ** 2)
            rdp_total = steps * rdp_per_step
            epsilon = rdp_total + math.log(1 / delta) / (alpha - 1)
            best_epsilon = min(best_epsilon, epsilon)

        return best_epsilon

    def _classify_privacy(self, epsilon: float) -> str:
        """Classify privacy level based on ε."""
        if epsilon < 1:
            return "strong_privacy (ε<1) → utility drops ~5-10%"
        elif epsilon < 5:
            return "moderate_privacy (ε<5) → utility drops ~3-5%"
        elif epsilon < 10:
            return "acceptable_privacy (ε<10) → utility drops ~1-3%"
        else:
            return "weak_privacy (ε>10) → minimal utility impact but weak guarantee"

    def model_privacy_utility_tradeoff(self) -> Dict:
        """Model privacy-utility tradeoff across different noise levels."""
        configs = [
            ("strong_privacy", 10.0, 10000),
            ("moderate_privacy", 3.0, 10000),
            ("acceptable_privacy", 1.0, 10000),
            ("minimal_privacy", 0.5, 10000),
            ("no_privacy", 0.01, 10000),
        ]

        results = {}
        for name, sigma, steps in configs:
            eps_result = self.compute_epsilon(sigma, steps)
            # Utility impact estimate (from literature)
            if eps_result["rdp_epsilon"] < 1:
                utility_pct = 90  # ~10% drop
            elif eps_result["rdp_epsilon"] < 5:
                utility_pct = 95  # ~5% drop
            elif eps_result["rdp_epsilon"] < 10:
                utility_pct = 97  # ~3% drop
            else:
                utility_pct = 99  # ~1% drop

            results[name] = {
                "epsilon": eps_result["rdp_epsilon"],
                "privacy_level": eps_result["privacy_level"],
                "utility_pct": utility_pct,
                "noise_multiplier": sigma,
            }

        return results


# ============================================================================
# Part 3: Transparency Scoring
# ============================================================================

class TransparencyScorer:
    """Score model transparency across documentation + interpretability dimensions.

    Key insight from ethics deep dive:
    → Transparency = 5 principles → documentation + interpretability + auditing
    → → Model cards: what model does + limitations + intended use → accountability!
    → → Interpretability: can we understand model decisions → mechanistic interpretability!
    → → → RTX 4090: activation patching → identify which neurons → practical interpretability!
    → → → → Auditing: external evaluation → red-team + bias testing → third-party trust!

    → LLM transparency special challenge:
    → → Large models → hard to interpret → "black box" concern → need mechanistic interp!
    → → → But: probing + activation patching → can identify circuits → progress!
    → → → → Anthropic: "Constructing a circuit" → mechanistic interpretability frontier!
    """

    # Transparency assessment dimensions
    DIMENSIONS = {
        "model_card": {
            "description": "Documentation: what model does, limitations, intended use",
            "max_score": 10,
            "sub_dimensions": {
                "intended_use": 2,
                "limitations": 2,
                "training_data": 2,
                "performance_metrics": 2,
                "ethical_considerations": 2,
            },
        },
        "interpretability": {
            "description": "Can we understand model decisions → probing + circuits",
            "max_score": 10,
            "sub_dimensions": {
                "probing_available": 3,
                "activation_patching": 3,
                "feature_steering": 2,
                "circuit_analysis": 2,
            },
        },
        "auditing": {
            "description": "External evaluation → red-team + bias testing",
            "max_score": 10,
            "sub_dimensions": {
                "red_team_testing": 3,
                "bias_audit": 3,
                "third_party_eval": 2,
                "safety_benchmark": 2,
            },
        },
        "data_documentation": {
            "description": "Training data provenance + curation documentation",
            "max_score": 10,
            "sub_dimensions": {
                "data_sources": 3,
                "curation_process": 3,
                "contamination_check": 2,
                "bias_analysis": 2,
            },
        },
        "deployment_transparency": {
            "description": "How model is deployed + monitored + updated",
            "max_score": 10,
            "sub_dimensions": {
                "deployment_documentation": 3,
                "monitoring": 2,
                "incident_response": 2,
                "update_policy": 3,
            },
        },
    }

    # Model transparency profiles
    MODEL_PROFILES = {
        "open_source_llm": {
            "model_card": 7,
            "interpretability": 4,
            "auditing": 5,
            "data_documentation": 3,
            "deployment_transparency": 4,
        },
        "proprietary_llm": {
            "model_card": 8,
            "interpretability": 2,
            "auditing": 6,
            "data_documentation": 2,
            "deployment_transparency": 5,
        },
        "rtx4090_local_model": {
            "model_card": 5,
            "interpretability": 6,  # can do probing locally!
            "auditing": 7,  # full local control → can audit!
            "data_documentation": 4,
            "deployment_transparency": 8,  # full control of deployment!
        },
    }

    def score_model(self, profile: str = "rtx4090_local_model") -> Dict:
        """Score a model's transparency."""
        if profile not in self.MODEL_PROFILES:
            profile = "rtx4090_local_model"

        scores = self.MODEL_PROFILES[profile]
        total = sum(scores.values())
        max_total = sum(d["max_score"] for d in self.DIMENSIONS.values())

        detailed = {}
        for dim, score in scores.items():
            pct = score / self.DIMENSIONS[dim]["max_score"] * 100
            detailed[dim] = {
                "score": score,
                "max": self.DIMENSIONS[dim]["max_score"],
                "percentage": pct,
                "description": self.DIMENSIONS[dim]["description"],
            }

        return {
            "profile": profile,
            "total_score": total,
            "max_score": max_total,
            "overall_percentage": total / max_total * 100,
            "dimensions": detailed,
            "interpretation": self._interpret_score(total / max_total * 100),
        }

    def _interpret_score(self, pct: float) -> str:
        """Interpret transparency score."""
        if pct >= 80:
            return "Excellent transparency → full documentation + interpretability + auditing"
        elif pct >= 60:
            return "Good transparency → most dimensions covered → some gaps"
        elif pct >= 40:
            return "Moderate transparency → significant gaps → need improvement"
        else:
            return "Poor transparency → major gaps → not trustworthy for deployment"


# ============================================================================
# Part 4: Governance Compliance — EU AI Act + ASL Risk
# ============================================================================

class GovernanceCompliance:
    """Check compliance with EU AI Act risk classification + ASL risk levels.

    Key insight from ethics deep dive:
    → EU AI Act: 4 risk levels → Unacceptable/High/Limited/Minimal → different obligations!
    → → Unacceptable: social scoring → banned!
    → → → High: hiring/credit/medical → strict compliance → documentation+audit+transparency!
    → → → → Limited: chatbots/content filtering → transparency obligation → disclose AI use!
    → → → → → Minimal: spam filters/games → no obligations → free!

    → ASL (Anthropic Responsible Scaling):
    → → ASL-1: low risk → no special measures → most current LLMs
    → → → ASL-2: medium risk → basic evaluation → GPT-4/Claude/DeepSeek-V3
    → → → → ASL-3: high risk → strict measures → frontier models → not yet
    → → → → → ASL-4: catastrophic risk → extreme measures → far future

    → → → → → Infra engineer = governance executor → implement technical controls!
    → → → → → → Guardrails=technical enforcement → <5% overhead → practical!
    """

    # EU AI Act use-case risk classification
    USE_CASE_RISKS = {
        "chatbot_general": {"eu_risk": "Limited", "asl_risk": "ASL-1",
                            "obligations": ["transparency: disclose AI use"]},
        "content_filtering": {"eu_risk": "Limited", "asl_risk": "ASL-1",
                             "obligations": ["transparency: disclose AI use"]},
        "hiring_assessment": {"eu_risk": "High", "asl_risk": "ASL-2",
                             "obligations": ["documentation", "audit", "transparency",
                                            "human oversight", "bias testing"]},
        "credit_scoring": {"eu_risk": "High", "asl_risk": "ASL-2",
                          "obligations": ["documentation", "audit", "transparency",
                                         "human oversight", "bias testing"]},
        "medical_diagnosis": {"eu_risk": "High", "asl_risk": "ASL-2",
                             "obligations": ["documentation", "audit", "transparency",
                                            "human oversight", "clinical validation"]},
        "social_scoring": {"eu_risk": "Unacceptable", "asl_risk": "ASL-3+",
                          "obligations": ["BANNED under EU AI Act!"]},
        "autonomous_weapon": {"eu_risk": "Unacceptable", "asl_risk": "ASL-4",
                             "obligations": ["BANNED! No deployment under any framework!"]},
        "code_generation": {"eu_risk": "Minimal", "asl_risk": "ASL-1",
                           "obligations": ["minimal: no specific obligations"]},
        "vlm_serving": {"eu_risk": "Limited", "asl_risk": "ASL-1",
                       "obligations": ["transparency: disclose AI use"]},
    }

    def classify_use_case(self, use_case: str) -> Dict:
        """Classify a use case under EU AI Act + ASL framework."""
        if use_case not in self.USE_CASE_RISKS:
            return {"error": f"Unknown use case: {use_case}"}

        classification = self.USE_CASE_RISKS[use_case]

        # Compute compliance requirements
        eu_level = classification["eu_risk"]
        obligations = classification["obligations"]

        # Compliance cost estimation (technical implementation)
        if eu_level == "Unacceptable":
            compliance_cost_pct = 0  # banned, can't deploy
        elif eu_level == "High":
            compliance_cost_pct = 15  # ~15% engineering overhead for compliance
        elif eu_level == "Limited":
            compliance_cost_pct = 3  # ~3% for transparency disclosures
        else:
            compliance_cost_pct = 0  # minimal obligations

        return {
            "use_case": use_case,
            "eu_ai_act_risk": eu_level,
            "asl_risk_level": classification["asl_risk"],
            "obligations": obligations,
            "compliance_cost_pct": compliance_cost_pct,
            "deployable": eu_level != "Unacceptable",
        }

    def assess_all_use_cases(self) -> Dict:
        """Assess all use cases."""
        results = {}
        for use_case in self.USE_CASE_RISKS:
            results[use_case] = self.classify_use_case(use_case)
        return results


# ============================================================================
# Main: Run all demonstrations
# ============================================================================

def main():
    print("=" * 70)
    print("AI Ethics & Governance Simulator — Fairness + Privacy + Transparency")
    print("=" * 70)
    print()

    # === Part 1: Fairness Assessment ===
    print("--- Part 1: Fairness Assessment (4 Definitions) ---")
    fairness = FairnessAssessment()
    assessment = fairness.full_assessment()

    for name, result in [
        ("DemographicParity", assessment["demographic_parity"]),
        ("EqualizedOdds", assessment["equalized_odds"]),
        ("Calibration", assessment["calibration"]),
        ("IndividualFairness", assessment["individual_fairness"]),
    ]:
        fair = "FAIR" if result["fair"] else "UNFAIR"
        gap = result.get("gap", result.get("tpr_gap", 0))
        print(f"  {name}: {fair} (gap={gap:.4f})")
        if name == "DemographicParity":
            for group, rate in result["positive_rates"].items():
                print(f"    {group}: positive rate = {rate:.3f}")
        elif name == "EqualizedOdds":
            for group in result["tpr_rates"]:
                print(f"    {group}: TPR={result['tpr_rates'][group]:.3f}, "
                      f"FPR={result['fpr_rates'][group]:.3f}")
        elif name == "Calibration":
            for group, data in result["per_group"].items():
                print(f"    {group}: avg calibration error = {data['avg_calibration_error']:.4f}")
    print()
    print(f"  Impossibility theorem: Cannot satisfy all 4 simultaneously!")
    print(f"  → Must choose context-appropriate definition → no universal answer!")
    print()

    # === Part 2: Privacy Modeling ===
    print("--- Part 2: Privacy Modeling (DP-SGD ε-Privacy) ---")
    privacy = PrivacyModel()

    tradeoff = privacy.model_privacy_utility_tradeoff()
    for name, data in tradeoff.items():
        print(f"  {name}: ε={data['epsilon']:.2f}, "
              f"utility={data['utility_pct']}%, "
              f"σ={data['noise_multiplier']}")
    print()

    # Specific config for LLM training
    eps_7b = privacy.compute_epsilon(noise_multiplier=3.0, num_steps=5000)
    print(f"  7B model training (σ=3.0, 5000 steps):")
    print(f"    ε (RDP) = {eps_7b['rdp_epsilon']:.2f}")
    print(f"    Level = {eps_7b['privacy_level']}")
    print()
    print("  Insight: ε<10 → acceptable privacy → ~3% utility drop → practical!")
    print()

    # === Part 3: Transparency Scoring ===
    print("--- Part 3: Transparency Scoring ---")
    scorer = TransparencyScorer()

    for profile in ["open_source_llm", "proprietary_llm", "rtx4090_local_model"]:
        result = scorer.score_model(profile)
        print(f"  {profile}: {result['overall_percentage']:.0f}% "
              f"({result['total_score']}/{result['max_score']}) → {result['interpretation']}")
        for dim, data in result["dimensions"].items():
            print(f"    {dim}: {data['score']}/{data['max']} ({data['percentage']:.0f}%)")
    print()
    print("  Insight: RTX 4090 local model = best interpretability+auditing → full local control!")
    print()

    # === Part 4: Governance Compliance ===
    print("--- Part 4: Governance Compliance (EU AI Act + ASL) ---")
    governance = GovernanceCompliance()
    all_cases = governance.assess_all_use_cases()

    for use_case, data in all_cases.items():
        deployable = "YES" if data["deployable"] else "BANNED"
        print(f"  {use_case}: EU={data['eu_ai_act_risk']}, ASL={data['asl_risk_level']}, "
              f"deployable={deployable}, cost={data['compliance_cost_pct']}%")
    print()
    print("  Insight: AI Infra engineer = governance executor → implement technical controls!")
    print("  → Guardrails <5% overhead → compliance achievable on RTX 4090!")
    print()

    # === Summary ===
    print("=" * 70)
    print("AI Ethics & Governance Summary:")
    print(f"  Fairness: 4 definitions conflict → choose context-appropriate → no universal!")
    print(f"  Privacy: DP-SGD ε-privacy → ε<10 → ~3% utility drop → practical!")
    print(f"  Transparency: RTX 4090 local = best → full control + local interpretability!")
    print(f"  Governance: EU AI Act → High risk=15% compliance cost → Limited=3% → manageable!")
    print()
    print("  Key lesson for AI Infra:")
    print("  → Ethics ≠ overhead → ethics = engineering → technical implementation!")
    print("  → Fairness is mathematical → measure → choose definition → implement!")
    print("  → Privacy is quantitative → ε budget → compose → track → guarantee!")
    print("  → → RTX 4090 = best platform for ethical AI → local control + low cost!")

    # Save results
    results = {
        "fairness_dp_fair": assessment["demographic_parity"]["fair"],
        "fairness_eo_fair": assessment["equalized_odds"]["fair"],
        "fairness_cal_fair": assessment["calibration"]["fair"],
        "fairness_ind_fair": assessment["individual_fairness"]["fair"],
        "privacy_epsilon_7b": eps_7b["rdp_epsilon"],
        "transparency_rtx4090_pct": scorer.score_model("rtx4090_local_model")["overall_percentage"],
        "governance_high_risk_cost_pct": 15,
    }
    with open("results/ai_ethics_governance_simulator.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/ai_ethics_governance_simulator.json")


if __name__ == "__main__":
    main()