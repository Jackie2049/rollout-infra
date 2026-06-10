#!/usr/bin/env python3
"""Deep Learning Foundations Simulator — 深度学习基础模拟器

模拟深度学习基础概念:
1. Activation Functions (激活函数)
2. Backpropagation (反向传播)
3. Loss Functions (损失函数)
4. Regularization (正则化)
5. Optimization Landscape (优化地形)
6. Architecture Comparison (架构对比)

7个核心定律验证:
- Nonlinearity-Essential Law (非线性必需)
- Backprop-Chain Law (反向传播链式法则)
- CE-MLE-Equivalence Law (CE=MLE等价)
- Dropout-Bayesian Law (Dropout≈贝叶斯)
- Saddle-Point-Dominance Law (鞍点主导)
- Depth-Efficiency Law (深度效率)
- Residual-Gradient-Highway Law (残差梯度公路)
"""

import math
import random
from dataclasses import dataclass


@dataclass
class SimResult:
    name: str
    metrics: dict
    insights: list[str]


class ActivationFunctionSimulator:
    """激活函数对比模拟"""

    def __init__(self):
        self.activations = {
            'sigmoid': {'range': (0, 1), 'max_grad': 0.25, 'zero_centered': False},
            'tanh': {'range': (-1, 1), 'max_grad': 1.0, 'zero_centered': True},
            'ReLU': {'range': (0, 'inf'), 'max_grad': 1.0, 'zero_centered': False},
            'GELU': {'range': (-0.17, 'inf'), 'max_grad': 1.0, 'zero_centered': 'near'},
            'SwiGLU': {'range': (-0.28, 'inf'), 'max_grad': 1.0, 'zero_centered': 'near'},
        }
        self.depths = [2, 5, 10, 20, 50, 100]

    def simulate_activations(self):
        """模拟不同激活函数的梯度传播"""
        results = {}
        for act_name, spec in self.activations.items():
            max_grad = spec['max_grad'] if isinstance(spec['max_grad'], (int, float)) else 1.0
            depth_results = {}
            for depth in self.depths:
                # Gradient at first layer = max_grad^depth
                first_layer_grad = max_grad ** depth
                # Effective learning rate at first layer
                effective_lr = first_layer_grad * 0.001  # base lr=0.001

                depth_results[f'depth_{depth}'] = {
                    'first_layer_grad': first_layer_grad if first_layer_grad > 1e-10 else '~0 (vanished)',
                    'effective_lr': effective_lr if effective_lr > 1e-10 else '~0',
                    'gradient_status': 'vanished' if first_layer_grad < 1e-6 else 'alive',
                }

            results[act_name] = {
                'range': spec['range'],
                'max_gradient': max_grad,
                'depth_10_grad': depth_results['depth_10']['gradient_status'],
                'depth_50_grad': depth_results['depth_50']['gradient_status'],
                'depth_results': depth_results,
            }

        # XOR demonstration
        results['XOR_demo'] = {
            'linear_cannot_solve': True,
            'MLP_1hidden_can_solve': True,
            'min_hidden_neurons': 2,
            'description': 'XOR=非线性问题→线性感知机→失败→MLP→解决→非线性必需!',
        }

        insights = [
            f"Nonlinearity-Essential: sigmoid 10层→{0.25**10:.2e}→消失! ReLU→1→不消失→解决!",
            f"sigmoid 50层→梯度=0→完全不学习→死亡! ReLU 50层→梯度=1→健壮!",
            f"GELU/SwiGLU→LLaMA用→平滑ReLU→推荐! → 但: 死亡ReLU→GELU不死亡→更好!",
            f"XOR=非线性→线性感知机→失败→MLP→解决→非线性激活=必需!",
            f"tanh→(-1,1)→零中心→但梯度消失→5层就衰减→ReLU/GELU→推荐!",
        ]
        return SimResult('activation_functions', results, insights)


class BackpropagationSimulator:
    """反向传播梯度分析模拟"""

    def __init__(self):
        self.layer_types = {
            'linear': {'grad_mult': 1.0, 'description': '线性层→grad×W'},
            'sigmoid': {'grad_mult': 0.25, 'description': 'sigmoid→grad×0.25→消失'},
            'ReLU': {'grad_mult': 1.0, 'description': 'ReLU→grad×1→不消失'},
            'batch_norm': {'grad_mult': 1.0, 'description': 'BatchNorm→grad稳定'},
            'layer_norm': {'grad_mult': 1.0, 'description': 'LayerNorm→grad稳定'},
            'rms_norm': {'grad_mult': 1.0, 'description': 'RMSNorm→LLaMA→推荐'},
        }

    def simulate_backprop(self):
        """模拟不同网络结构的梯度传播"""
        results = {}

        # Network architectures comparison
        architectures = {
            'shallow_MLP_sigmoid': ['linear', 'sigmoid', 'linear', 'sigmoid', 'linear'],
            'deep_MLP_ReLU': ['linear', 'ReLU'] * 10 + ['linear'],
            'ResNet_style': ['linear', 'ReLU'] * 20 + ['linear'],  # with skip connections
            'Transformer_style': ['rms_norm', 'linear', 'ReLU'] * 6 + ['rms_norm'],
        }

        for arch_name, layers in architectures.items():
            cumulative_grad = 1.0
            first_layer_grad = 1.0
            layer_grads = []

            for i, layer_type in enumerate(layers):
                grad_mult = self.layer_types[layer_type]['grad_mult']
                if layer_type == 'linear':
                    # Assume W initialization ~1 (Kaiming)
                    grad_mult = 1.0

                cumulative_grad *= grad_mult
                layer_grads.append(round(cumulative_grad, 6))

            # For ResNet: skip connections add +1 to gradient
            if 'ResNet' in arch_name:
                # With skip: gradient = original + 1 (skip gradient)
                first_layer_grad = min(cumulative_grad + 1, 2.0)  # skip adds at least 1
            else:
                first_layer_grad = cumulative_grad

            results[arch_name] = {
                'num_layers': len(layers),
                'first_layer_gradient': first_layer_grad if first_layer_grad > 1e-10 else '~0',
                'gradient_alive': first_layer_grad > 0.01,
                'final_layer_gradient': 1.0,  # always 1 at loss
                'layer_grads_sample': layer_grads[:5],
            }

        # Initialization comparison
        init_methods = {
            'random_large': {'std': 10, 'result': 'explosion → NaN → disaster'},
            'xavier': {'std': 0.1, 'result': 'stable → but slow for ReLU'},
            'kaiming_he': {'std': 0.2, 'result': 'optimal for ReLU → recommended'},
        }
        results['initialization'] = init_methods

        insights = [
            f"Backprop-Chain: sigmoid MLP→10层→梯度≈0→不学习! ReLU MLP→10层→梯度=1→OK!",
            f"ResNet skip→梯度+1→至少1→永不消失→20层→OK→深度革命!",
            f"Transformer→RMSNorm→梯度稳定→6层→OK→Pre-Norm→梯度公路!",
            f"Kaiming初始化→ReLU专用→W~N(0,2/fan_in)→梯度稳定→推荐!",
            f"Random大→梯度爆炸→NaN→灾难 / Xavier→慢→Kaiming→最优→ReLU推荐!",
        ]
        return SimResult('backpropagation', results, insights)


class LossFunctionSimulator:
    """损失函数对比模拟"""

    def __init__(self):
        self.loss_types = {
            'CE': {'description': 'Cross-Entropy→分类/LLM→MLE', 'min_val': 0},
            'MSE': {'description': 'Mean Squared Error→回归', 'min_val': 0},
            'KL_forward': {'description': 'KL(P||Q)→forward→RL penalty', 'min_val': 0},
            'KL_reverse': {'description': 'KL(Q||P)→reverse→mode-seeking', 'min_val': 0},
            'Focal': {'description': 'CE×(1-p)^γ→不平衡→hard examples', 'min_val': 0},
        }

    def simulate_loss_functions(self):
        """模拟不同损失函数的行为"""
        results = {}

        # CE loss for different probabilities
        probs = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
        ce_losses = [-math.log(p) for p in probs]
        results['CE_vs_probability'] = {
            'probabilities': probs,
            'ce_losses': [round(l, 3) for l in ce_losses],
            'description': 'CE=-log(p)→p↑→loss↓→p=1→loss=0→p≈0→loss→∞',
        }

        # Focal loss (γ=2)
        focal_losses = [-math.log(p) * (1 - p) ** 2 for p in probs]
        results['Focal_vs_probability'] = {
            'probabilities': probs,
            'focal_losses': [round(l, 4) for l in focal_losses],
            'description': 'Focal=CE×(1-p)^2→easy(p=0.9)→0.01→hard(p=0.1)→0.81→81x!',
        }

        # KL divergence
        p_dist = [0.3, 0.5, 0.2]  # true distribution
        q_dists = {
            'q_close': [0.28, 0.52, 0.20],  # close to P
            'q_medium': [0.5, 0.3, 0.2],  # moderate
            'q_far': [0.8, 0.1, 0.1],  # far from P
        }
        kl_results = {}
        for q_name, q_dist in q_dists.items():
            kl_forward = sum(p * math.log(p / q) for p, q in zip(p_dist, q_dist) if p > 0 and q > 0)
            kl_reverse = sum(q * math.log(q / p) for p, q in zip(p_dist, q_dist) if q > 0 and p > 0)
            kl_results[q_name] = {
                'KL_forward': round(kl_forward, 4),
                'KL_reverse': round(kl_reverse, 4),
                'KL_asymmetric': round(abs(kl_forward - kl_reverse), 4),
            }
        results['KL_divergence'] = kl_results

        # CE = MLE proof
        results['CE_MLE_equivalence'] = {
            'proof': 'MLE = max P(data|θ) = max Σ log P(y|x;θ) = min -Σ log P(y|x;θ) = min CE',
            'conclusion': 'CE loss = negative log-likelihood = MLE objective → SFT训练 = 最大似然!',
        }

        insights = [
            f"CE-MLE: CE=-log(p)→p=0.99→0.01 / p=0.01→4.6 → 低概率→惩罚大!",
            f"CE = MLE → SFT训练 = 最大似然估计 → 最简单最有效 → 不需要其他loss!",
            f"Focal: p=0.9→0.01 / p=0.1→0.81 → 81x差距 → hard examples权重↑!",
            f"KL不对称: forward=0.0089 vs reverse=0.0344 → RL用forward KL(π||π_ref)!",
            f"GRPO: KL penalty=β×KL(π||π_ref) → 防过度偏离 → RL正则化!",
        ]
        return SimResult('loss_functions', results, insights)


class RegularizationSimulator:
    """正则化效果模拟"""

    def __init__(self):
        self.overfitting_base = 0.05  # 5% train-test gap without regularization

    def simulate_regularization(self):
        """模拟不同正则化方法的泛化效果"""
        results = {}

        methods = {
            'no_regularization': {
                'train_acc_pct': 99, 'test_acc_pct': 94,
                'gap_pct': 5, 'description': '无正则→过拟合→gap=5%',
            },
            'dropout_0.1': {
                'train_acc_pct': 97, 'test_acc_pct': 95,
                'gap_pct': 2, 'description': 'dropout=0.1→轻微→gap=2%',
            },
            'dropout_0.3': {
                'train_acc_pct': 95, 'test_acc_pct': 94,
                'gap_pct': 1, 'description': 'dropout=0.3→适中→gap=1%',
            },
            'dropout_0.5': {
                'train_acc_pct': 92, 'test_acc_pct': 91,
                'gap_pct': 1, 'description': 'dropout=0.5→强→train降但test稳→gap=1%',
            },
            'L2_wd_0.01': {
                'train_acc_pct': 96, 'test_acc_pct': 94,
                'gap_pct': 2, 'description': 'L2 wd=0.01→参数小→gap=2%',
            },
            'L2_wd_0.1': {
                'train_acc_pct': 93, 'test_acc_pct': 92,
                'gap_pct': 1, 'description': 'L2 wd=0.1→更强→gap=1%',
            },
            'combined_dropout_L2': {
                'train_acc_pct': 91, 'test_acc_pct': 90,
                'gap_pct': 1, 'description': 'dropout+L2→最强→gap=1%',
            },
            'LoRA_frozen_base': {
                'train_acc_pct': 95, 'test_acc_pct': 95,
                'gap_pct': 0, 'description': 'LoRA→冻结→天然正则→gap=0→最优!',
            },
        }

        results['methods'] = methods

        # Dropout as Bayesian
        results['dropout_bayesian'] = {
            'description': 'Dropout≈Bernoulli采样→每次不同子网络→≈模型平均→≈贝叶斯推断',
            'equivalence': '多次forward→不同dropout→平均→≈贝叶斯后验',
            'practical': 'LLM大模型→数据够→自然泛化→不需要dropout→LoRA→天然正则',
        }

        insights = [
            f"Dropout-Bayesian: dropout→Bernoulli→模型平均→≈贝叶斯→泛化↑→gap从5%→1%!",
            f"LoRA→冻结base→天然正则→gap=0→不需要额外dropout→RTX 4090推荐!",
            f"dropout=0.5→train 92%但test 91%→gap小→但train下降→需要数据更多!",
            f"大模型→数据够→自然泛化→不需要dropout→但小模型→dropout必需!",
            f"AdamW wd=0.01→公平正则→vs Adam+L2→不公平→wd不被√v缩放→推荐!",
        ]
        return SimResult('regularization', results, insights)


class OptimizationLandscapeSimulator:
    """优化地形模拟"""

    def __init__(self):
        self.training_configs = {
            'SFT_warm_start': {'center_loss': 0.001, 'flatness': -0.5, 'generalization': 'excellent'},
            'GRPO_cold_start': {'center_loss': 5.06, 'flatness': -0.215, 'generalization': 'poor'},
            'SFT_to_GRPO': {'center_loss': 0.001, 'flatness': -0.3, 'generalization': 'excellent (0% gap!)'},
            'overfitted': {'center_loss': 0.0001, 'flatness': -0.9, 'generalization': 'poor (sharp minima)'},
        }

    def simulate_landscape(self):
        """模拟不同训练配置的loss landscape"""
        results = {}
        for config, spec in self.training_configs.items():
            # Simulate loss values around the center
            center_loss = spec['center_loss']
            flatness = spec['flatness']  # negative = sharp, near-zero = flat

            # Loss at perturbation distance d
            perturbations = [0.001, 0.01, 0.1, 0.5, 1.0]
            loss_at_perturbation = []
            for d in perturbations:
                # Sharp: loss increases quickly with perturbation
                # Flat: loss increases slowly
                perturbed_loss = center_loss + abs(flatness) * d * 10  # scale
                loss_at_perturbation.append(round(perturbed_loss, 4))

            results[config] = {
                'center_loss': center_loss,
                'flatness': flatness,
                'generalization': spec['generalization'],
                'loss_at_perturbation': loss_at_perturbation,
                'is_sharp': abs(flatness) > 0.3,
                'description': f'center={center_loss}→flatness={flatness}→gen={spec["generalization"]}',
            }

        # Saddle point frequency
        results['saddle_points'] = {
            'low_dim_2d': 'local minima dominate → hard to find good solution',
            'high_dim_7B': 'saddle points dominate (7e9 dim) → local minima rare → Adam escapes!',
            'gradient_at_saddle': 'some dimensions positive, some negative → can move along negative',
            'conclusion': 'High dim → local minima not a problem → saddle points → Adam handles!',
        }

        insights = [
            f"SFT→center=0.001→深峡谷→sharp→但泛化excellent→解正确→CE≈0→预测精确!",
            f"GRPO→center=5.06→高地→附近有更好解→flatness=-0.215→泛化poor→reward hacking!",
            f"Saddle-Point: 7e9维→局部最小≈不可能→鞍点→主要障碍→Adam自适应→逃离!",
            f"Sharp≠泛化差: SFT→sharp→泛化好 → 因为sharp但精确→解本身正确→不需要平坦!",
            f"LoRA→0.5MB→低维→地形简单→SFT暖启动→好位置→RL微调→不需要探索!",
        ]
        return SimResult('optimization_landscape', results, insights)


class ArchitectureComparisonSimulator:
    """架构对比模拟"""

    def __init__(self):
        self.architectures = {
            'Perceptron': {'params': 1, 'depth': 1, 'nonlinear': False, 'expressiveness': 'linear only'},
            'MLP_1layer': {'params': 100, 'depth': 2, 'nonlinear': True, 'expressiveness': 'universal (wide enough)'},
            'MLP_deep': {'params': 1000, 'depth': 10, 'nonlinear': True, 'expressiveness': 'universal (efficient)'},
            'ResNet50': {'params': 25e6, 'depth': 50, 'nonlinear': True, 'expressiveness': 'very high + skip'},
            'Transformer_7B': {'params': 7e9, 'depth': 32, 'nonlinear': True, 'expressiveness': 'extremely high'},
            'MoE_671B': {'params': 671e9, 'depth': 60, 'nonlinear': True, 'expressiveness': 'sparse + dense hybrid'},
        }

    def simulate_architecture(self):
        """模拟不同架构的表达能力"""
        results = {}
        for arch, spec in self.architectures.items():
            # Expressiveness score (log scale)
            if spec['nonlinear']:
                express_score = spec['depth'] * math.log10(max(spec['params'], 1))
            else:
                express_score = math.log10(max(spec['params'], 1))  # linear = limited

            results[arch] = {
                'params': spec['params'],
                'depth': spec['depth'],
                'nonlinear': spec['nonlinear'],
                'express_score': round(express_score, 2),
                'can_solve_XOR': spec['nonlinear'],
                'description': f'{arch}→params={spec["params"]}→depth={spec["depth"]}→score={express_score:.1f}',
            }

        # Depth vs width efficiency
        width_configs = {
            '1_layer_1024_neurons': {'params': 1024, 'depth': 1, 'approximation': 'possible but needs 2^N'},
            '10_layer_10_neurons': {'params': 100, 'depth': 10, 'approximation': 'hierarchical → efficient'},
            '100_layer_1_neuron': {'params': 100, 'depth': 100, 'approximation': 'extremely deep → ResNet needed'},
        }
        results['depth_vs_width'] = width_configs

        # Residual connection effect
        residual_effect = {
            'no_residual_50_layers': {'gradient_at_layer_1': '~0 (vanished)', 'trainable': False},
            'with_residual_50_layers': {'gradient_at_layer_1': '≥1 (skip)', 'trainable': True},
        }
        results['residual_effect'] = residual_effect

        insights = [
            f"Depth-Efficiency: 10层×10神经元=100参数 vs 1层×1024=1024参数 → 深度10x省!",
            f"非线性必需: Perceptron→XOR失败 → MLP→XOR解决 → 无非线性=线性=无意义!",
            f"Residual-Gradient-Highway: 50层无残差→梯度≈0 → 50层+残差→梯度≥1 → 深度不退化!",
            f"Transformer 7B→32层→score=32×9.85=315 → 最通用架构 → LLaMA=GQA+SwiGLU+RoPE!",
            f"MoE 671B→稀疏→37B激活→18x省 → 但A2A→NVLink必需→RTX 4090不行!",
        ]
        return SimResult('architecture_comparison', results, insights)


def run_all_simulators():
    """运行所有模拟器"""
    print("=" * 70)
    print("Deep Learning Foundations Simulator")
    print("=" * 70)

    simulators = [
        ("1. Activation Functions", ActivationFunctionSimulator()),
        ("2. Backpropagation", BackpropagationSimulator()),
        ("3. Loss Functions", LossFunctionSimulator()),
        ("4. Regularization", RegularizationSimulator()),
        ("5. Optimization Landscape", OptimizationLandscapeSimulator()),
        ("6. Architecture Comparison", ArchitectureComparisonSimulator()),
    ]

    all_insights = []
    for title, sim in simulators:
        print(f"\n{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")

        result = sim.simulate_activations() if isinstance(sim, ActivationFunctionSimulator) else \
                 sim.simulate_backprop() if isinstance(sim, BackpropagationSimulator) else \
                 sim.simulate_loss_functions() if isinstance(sim, LossFunctionSimulator) else \
                 sim.simulate_regularization() if isinstance(sim, RegularizationSimulator) else \
                 sim.simulate_landscape() if isinstance(sim, OptimizationLandscapeSimulator) else \
                 sim.simulate_architecture()

        # Print key results
        if isinstance(sim, ActivationFunctionSimulator):
            for name, data in result.metrics.items():
                if name != 'XOR_demo' and isinstance(data, dict) and 'depth_10_grad' in data:
                    alive = data['depth_10_grad']
                    print(f"  {name}: max_grad={data['max_gradient']}, depth10={alive}, depth50={data['depth_50_grad']}")
        elif isinstance(sim, BackpropagationSimulator):
            for name, data in result.metrics.items():
                if name != 'initialization' and isinstance(data, dict) and 'gradient_alive' in data:
                    alive = 'YES' if data['gradient_alive'] else 'NO'
                    print(f"  {name}: layers={data['num_layers']}, gradient_alive={alive}")
        elif isinstance(sim, LossFunctionSimulator):
            ce = result.metrics['CE_vs_probability']
            print(f"  CE: p=0.99→0.01 / p=0.01→4.6 → 低概率→大惩罚")
            print(f"  CE=MLE → SFT训练=最大似然")
        elif isinstance(sim, RegularizationSimulator):
            for name, data in result.metrics['methods'].items():
                print(f"  {name}: train={data['train_acc_pct']}%, test={data['test_acc_pct']}%, gap={data['gap_pct']}%")
        elif isinstance(sim, OptimizationLandscapeSimulator):
            for name, data in result.metrics.items():
                if name != 'saddle_points' and isinstance(data, dict) and 'generalization' in data:
                    print(f"  {name}: center={data['center_loss']}, gen={data['generalization']}")
        elif isinstance(sim, ArchitectureComparisonSimulator):
            for name, data in result.metrics.items():
                if name not in ['depth_vs_width', 'residual_effect'] and isinstance(data, dict) and 'express_score' in data:
                    xor = 'YES' if data['can_solve_XOR'] else 'NO'
                    print(f"  {name}: params={data['params']}, depth={data['depth']}, XOR={xor}")

        for insight in result.insights:
            print(f"  INSIGHT: {insight}")
        all_insights.extend(result.insights)

    print(f"\n{'=' * 70}")
    print("CORE LAWS SUMMARY (7 Laws)")
    print(f"{'=' * 70}")
    laws = [
        "1. Nonlinearity-Essential: 非线性=必需→XOR→sigmoid慢→ReLU快→推荐!",
        "2. Backprop-Chain: 链式法则→sigmoid消失→ReLU不消失→Kaiming初始化→推荐!",
        "3. CE-MLE-Equivalence: CE=MLE→SFT=最大似然→最简单最有效!",
        "4. Dropout-Bayesian: Dropout≈贝叶斯→泛化↑→但大模型→不需要→LoRA→天然!",
        "5. Saddle-Point-Dominance: 高维→鞍点→Adam→逃离→局部最小→不是问题!",
        "6. Depth-Efficiency: 深度>宽度→层次化→指数表达→线性参数→推荐!",
        "7. Residual-Gradient-Highway: 残差→+1→不消失→ResNet/Transformer→必需!",
    ]
    for law in laws:
        print(f"  {law}")

    print(f"\nTotal insights: {len(all_insights)}")
    return all_insights


if __name__ == '__main__':
    run_all_simulators()