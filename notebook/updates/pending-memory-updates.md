# Pending MEMORY.md Updates
# Add these to MEMORY.md when file becomes accessible

- **Chinchilla Scaling Laws**: L(N,D)=A/N^α+B/D^β+E, α=0.34, β=0.28, E=1.69, ~20 tok/param最优, LLaMA过训练推理更高效
- **Training Techniques RTX 4090**: AdamW>>Adam(L2)(4.13vs6.57), BF16省29%内存, grad_clip=1.0标准
- **PPO Training RTX 4090**: PPO比Vanilla PG好24%, KL β=0.01最优, epochs=4标准
- **KV Cache压缩 RTX 4090**: INT8 cos=0.9999安全, SW-4K节省97%(17GB→0.54GB for 7B/128K)
- **Mamba**: 选择性SSM(B,C,Δ=f(x)), 线性时间, 2.8B匹配GPT-J-6B, 推理5x快(无KV cache)
