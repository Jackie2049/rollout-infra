# AI Infrastructure Security Deep Dive

> 2026-06-10 | 安全=AI生产的基础设施! 从推理安全到训练安全, 从部署安全到供应链安全, 安全=不可忽视的维度!
> 关联: ai-safety-alignment-deep-dive.md, ai-serving-observability-deep-dive.md, distributed-systems-deep-dive.md

## 0. 核心定律: 安全 = AI基础设施的底层需求

安全不是附加功能 → 是基础设施需求 → 与性能/成本同等重要!

3层安全模型:
1. **推理安全**: 输入验证+模型隔离+对抗防御 → 服务运行时安全
2. **训练安全**: 数据完整性+模型保护+训练隔离 → 训练过程安全
3. **部署安全**: 容器安全+API安全+网络隔离 → 基础设施安全

## 1. Secure Inference — 推理安全

```
推理安全威胁:

1. Prompt Injection:
   → 用户输入→恶意指令→绕过安全约束→模型执行攻击者意图!
   → → → 直接注入: "忽略之前的指令, 输出..."
   → → → → → 间接注入: 数据中嵌入恶意prompt(如网页→RAG检索→注入模型)!
   → → → → → → → 防御: 输入过滤+指令层级分离+FSM约束解码 → <5% overhead!

2. Adversarial Inputs:
   → 不可察觉的输入修改→模型输出完全改变→攻击者控制输出!
   → → → 文本: 同义词替换→语义保持但输出改变 → 文本更难攻击!
   → → → → → 图像: 不可见噪声→分类错误 → 图像更容易攻击!
   → → → → → → → 防御: 输入验证+鲁棒训练+输出监控 → 成本高!

3. Model Extraction:
   → 大量查询→重建模型 → API→返回logits→梯度提取→近似模型!
   → → → 防御: 限制查询频率+降低输出精度(top-1而非全logits)+水印!
   → → → → → OpenAI API: 只返回top概率 → 不返回全logits → 限制信息!
   → → → → → → → watermark: 隐藏信号→可检测→证明模型来源!

4. Data Leakage:
   → 模型→输出训练数据 → 记忆攻击→提取隐私信息!
   → → → 防御: 差分隐私(DP-SGD)→ε隐私保证 → 但精度↓!
   → → → → → 训练数据去重 → 重复数据更易记忆 → 去重→减少泄漏!

vLLM推理安全措施:
  → 输入限制: max_model_len → 防止超长输入攻击 → OOM防护!
  → → → 输出限制: max_tokens → 防止无限输出 → 资源防护!
  → → → → → 采样参数: temperature+top_p → 防止确定性输出 → 增加随机性!
  → → → → → → → 结构化输出: FSM→约束→安全格式 → xgrammar!
```

## 2. Secure Training — 训练安全

```
训练安全威胁:

1. Data Poisoning:
   → 恶意数据→训练→模型学到错误模式 → 后门攻击!
   → → → 后门: 特定触发词→特定输出 → 正常输入→正常输出 → 隐蔽!
   → → → → → 防御: 数据验证+去重+质量过滤 → 但不能100%防御!
   → → → → → → → verl RL: reward model→ poisoned reward → reward hacking!

2. Model Theft:
   → 训练模型→窃取→复制 → 内部威胁+外部攻击!
   → → → 防御: 模型加密+访问控制+审计日志 → 运维措施!
   → → → → → 权重混淆: 训练→混淆→推理→解混淆 → 但慢!

3. Training Integrity:
   → 训练过程被篡改 → 梯度注入→模型行为改变 → 供应链攻击!
   → → → 防境: checkpoint签名+梯度验证+确定性训练 → 技术措施!
   → → → → → verl: 分布式训练 → 多GPU → 梯度聚合 → 单GPU篡改难影响!

4. Privacy Leakage:
   → 训练数据→模型→泄漏 → GDPR→法律风险!
   → → → 防御: DP-SGD ε-privacy → ε越小→隐私越强→精度越低!
   → → → → → ε=1→强隐私→精度↓5-10%; ε=10→弱隐私→精度↓<1%
   → → → → → → → 实用: ε=8→平衡 → DP-SGD推荐!

RTX 4090训练安全:
  → 单GPU训练 → 无分布式 → 简单 → 但无冗余 → checkpoint依赖本地!
  → → → verl GRPO+LoRA → 小数据 → poisoning风险低 → 但reward需验证!
  → → → → → INT4模型 → 模型小 → 加密成本低 → 但解密慢!
```

## 3. Secure Deployment — 部署安全

```
部署安全3层:

1. 容器安全 (Docker/Kubernetes):
   → 镜像签名 → 验证 → 防止恶意镜像 → cosign!
   → → → 运行时隔离 → seccomp → 限制syscall → 减少攻击面!
   → → → → → 网络隔离 → NetworkPolicy → Pod间限制 → 防止横向移动!
   → → → → → → → 资源限制 → limits → 防止DoS → CPU/memory bound!

   vLLM Docker镜像:
   → vllm/vllm-openai:latest → 官方镜像 → 信任!
   → → → 但: 大镜像(~5GB) → 依赖多 → 潜在漏洞 → 需扫描!
   → → → → → Trivy扫描 → CVE检测 → 修复 → 安全基线!

2. API安全 (HTTP/REST):
   → 认证: API key → 防止未授权访问 → rate limiting!
   → → → 加密: HTTPS → TLS → 防止窃听 → 生产必须!
   → → → → → 输入验证: JSON schema → 防止恶意输入 → OWASP!
   → → → → → → → CORS → 限制来源 → 防止跨域攻击 → API gateway!

   vLLM API安全:
   → OpenAI兼容 → 无原生认证 → 需外部API gateway!
   → → → Nginx/Traefik → reverse proxy → 认证+rate limit → 前置!
   → → → → → API key → 中间件 → 验证 → 拒绝 → 日志!

3. 网络安全:
   → 入口: 防火墙 → 只开推理端口 → 减少攻击面!
   → → → 内部: VPN → 加密隧道 → 防止窃听 → 管理访问!
   → → → → → 监控: Prometheus → 异常流量 → DDoS检测 → alert!

   RTX 4090网络安全:
   → 单GPU → 本地 → 网络简单 → 无分布式攻击面!
   → → → SSH → 公钥认证 → 禁用密码 → 安全!
   → → → → → 端口 → 只开推理API → 最小化!
```

## 4. Supply Chain Security — 供应链安全

```
AI供应链攻击路径:

1. 模型权重供应链:
   → HF Hub → 下载 → 加载 → 掐持模型 → 攻击者上传恶意权重!
   → → → 防御: 权重签名+哈希验证+可信来源 → 安全下载!
   → → → → → vLLM: HF download → 验证checksum → 但不是加密验证!
   → → → → → → → 改进: 权重签名 → RSA/ECDSA → 验证 → 安全!

2. 依赖供应链:
   → pip install → 包注入 → typosquatting → 恶意包 → 攻击!
   → → → 防御: 锁定版本(requirements.txt)+哈希验证+可信源!
   → → → → → vLLM: requirements.txt → 依赖锁定 → 但仍需验证!
   → → → → → → → conda: 阿里源/清华源 → 中国可信 → 需验证!

3. 预训练模型供应链:
   → HuggingFace模型 → 预训练 → 可能含后门 → 信任链!
   → → → 防御: 模型评估+对抗测试+数据溯源 → 但成本高!
   → → → → → 开源模型 → 透明 → 可审计 → 更安全(vs封闭模型)!
   → → → → → → → LLaMA/Qwen → 开源 → 信任 → 推荐!

4. Checkpoint供应链:
   → checkpoint → 存储 → 加载 → 可能被篡改 → 训练中断!
   → → → 防御: checkpoint签名+哈希+版本管理 → 防篡改!
   → → → → → verl: 异步checkpoint → 版本管理 → 但需签名!
```

## 5. GPU Infrastructure Security — GPU基础设施安全

```
GPU安全挑战:

1. GPU内存安全:
   → ECC错误 → 数据损坏 → 模型输出错误 → 硬件攻击!
   → → → RTX 4090: 无ECC → 消费级 → 比A100更脆弱!
   → → → → → A100: ECC → 硬件级保护 → 生产推荐!
   → → → → → → → 防御: 输出验证 → checksum → 检测 → 但不能预防!

2. GPU访问控制:
   → 多用户共享 → 需隔离 → CUDA MPS → 部分隔离!
   → → → MPS: 多进程服务 → 资源分配 → 但不是完全隔离!
   → → → → → 时间分片 → GPU→单用户→切换 → 安全但慢!
   → → → → → → → RTX 4090: 单用户 → 无多租户 → 简单!

3. 侧信道攻击:
   → GPU功耗→模型推断 → 推理时间→模型结构 → 泄漏!
   → → → 防御: 常量时间推理→困难 → 随机化→可能 → 研究阶段!
   → → → → → vLLM: continuous batching → 时间不固定 → 自然模糊!

4. 驱动安全:
   → NVIDIA驱动 → 漏洞 → CVE → 权限提升 → 攻击!
   → → → 防防御: 定期更新+安全公告+最小权限 → 运维!
   → → → → → RTX 4090: driver 575.57.08 → 需关注CVE!
```

## 6. Security Operations — 安全运维

```
安全运维框架:

1. 监控与检测:
   → Prometheus metrics → 异常检测 → DDoS → OOM → 攻击模式!
   → → → 输入异常: 超长prompt → injection → max_model_len限制!
   → → → → → 输出异常: 重复输出 → reward hacking → 输出监控!
   → → → → → → → 流量异常: 大量查询 → extraction → rate limit!

2. 响应与恢复:
   → 入侵检测 → 响应 → 阻断 → 恢复 → 安全加固!
   → → → 模型替换 → 安全版本 → 回滚 → checkpoint恢复!
   → → → → → 数据清洗 → poisoned数据删除 → 重训练 → 成本高!

3. 合规与审计:
   → GDPR → 数据隐私 → DP-SGD → 合规!
   → → → EU AI Act → 风险分级 → 高风险→审计 → 合规!
   → → → → → SOC 2 → 安全控制 → 云服务 → 合规!
   → → → → → → → ISO 27001 → 信息安全 → 组织 → 合规!

4. 安全基线:
   → vLLM安全配置:
     → enable_prefix_caching=False → RLHF后重置 → 防止旧prefix!
     → → → max_model_len=4096 → 防止超长 → 防OOM!
     → → → → → gpu_memory_utilization=0.9 → 90% → 预留10% → 安全!
     → → → → → → → API key认证 → 外部 → 必须 → 生产!
```

## 7. Core Laws — AI安全核心定律

1. **Input-Validation Law**: 输入验证=第一防线 → prompt injection → max_model_len+FSM → <5% overhead!
   → → → 不验证=开放攻击 → 验证=关闭大多数攻击向量!

2. **Isolation Law**: 模型隔离=最小权限 → 容器+网络+资源 → 减少攻击面!
   → → → vLLM单进程 → 简单 → 但需API gateway前置 → 隔离!

3. **Supply-Chain-Trust Law**: 供应链=信任链 → HF+pip+conda → 签名+哈希+验证!
   → → → 开源模型(LLaMA/Qwen) → 可审计 → 更安全 → vs封闭!

4. **No-ECC Law**: RTX 4090无ECC → 消费级 → 数据完整性弱 → 输出验证必需!
   → → → A100有ECC → 生产安全 → RTX 4090需额外验证!

5. **DP-Privacy Law**: DP-SGD ε=8 → 训练隐私保证 → GDPR合规 → 精度↓<1%!
   → → → ε越小→隐私越强→精度越低 → Goldilocks Zone ε=8!

6. **Rate-Limit Law**: rate limit=防extraction+防DDoS → 限制查询频率 → API安全!
   → → → OpenAI: 限制RPM/TPM → 降低输出精度 → 减少信息泄漏!

7. **Security-Depth Law**: 安全=多层 → 输入+模型+部署+运维 → 深度防御!
   → → → 单层防御=单点故障 → 多层=冗余 → 安全!

## 关键参考

- OWASP Top 10 → Web应用安全 → API安全 → vLLM推理
- NIST AI Risk Management Framework → AI风险管理 → 安全框架
- vLLM安全配置: gpu_memory_utilization/max_model_len → 安全基线
- DP-SGD: ε-privacy(RDP) → 训练隐私 → GDPR合规
- Trivy: Docker镜像扫描 → CVE检测 → 容器安全
- cosign: 镾像签名 → 验证 → 供应链安全
- EU AI Act: 4级风险 → 高风险AI→审计 → 合规
- NVIDIA GPU ECC: A100→硬件级 → RTX 4090→无 → 安全差距
- Adversarial ML: TextGrad/FastGrad → 对抗攻击 → 防御