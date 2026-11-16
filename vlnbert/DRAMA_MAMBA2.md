# Memoir 的 Drama / Mamba-2 世界模型迁移

实现是 Drama 启发的 VLN 世界模型后端，**不是原 Atari Drama 的完整复现**。
选择其默认两层 Mamba-2、d_model=512、d_state=16、expand=2、headdim=64、
causal depthwise conv width=4。保持 Memoir 的 Gaussian stochastic=96、
deterministic=672、最终 768 维 latent、语言条件、KL/overshooting 与重建接口。

差异明确如下：

- 原 Drama 使用图像 CNN 和 32×32 categorical latent、Atari actor/replay；
  本迁移使用冻结 Memoir 的语言/全景编码器及连续 latent，不复制其 actor 或采样器。
- `world_model_type='tssm'` 仅是作者代码的序列输入/缓存调用约定；真正后端由
  `config.world_model_backend='drama_mamba2'` 选择，prior 内没有时间 Transformer。
- 语言条件使用全 instruction token 的带 mask cross-attention；instruction 是已知条件，
  时间动力学本身严格因果。原 Memoir prior 的输入是上一 latent，并无单独 Atari
  action ID 接口；保留原 latent 和 reward/action reconstruction 调用语义。
- 使用可微、无 in-place 更新的 PyTorch Mamba-2 recurrence。它包含 input-dependent
  dt/B/C、每 head 的连续 A、skip D、causal convolution 和 gated RMSNorm，**不是 GRU**。
  未使用融合 CUDA/Triton kernel，不能直接引用官方融合实现的速度数据。
- 官方 Mamba2 inference step 会原地修改 cache，不能直接用于本文训练中的多分支
  overshooting。这里 cache 全部以新 tensor 返回，训练时不 detach 旧 state；可反传至历史。
- 为兼容作者四槽缓存 stack/unstack，每层是 `(conv_state, ssm_state, language_keys,
  language_values)`。前两槽是固定尺寸的状态，不随时间增长；后两槽属于当前指令，
  新 rollout 必须从 None 开始，不能跨不同指令复用。

## 构造和冻结

给 `vlnbert.vlnbert_init.get_vlnbert_models(args)` 增加：

```python
args.world_model_backend = 'drama_mamba2'
args.freeze_encoders = True
args.drama_d_model = 512
args.drama_layers = 2
args.drama_d_state = 16
args.drama_headdim = 64
args.drama_dropout = 0.1
```

从作者 TSSM 初始化权重新建预训练时还必须显式设
`args.initialize_drama_world=True`：过滤旧 world-model 参数，随机初始化新模型；
导航/encoder 参数仍按原作者加载逻辑导入。没有此标记且权重缺少
`prior.layers.0.*` 时会报错，避免把 IR2R 的缺失预训练误当成自动从头训练。
调用者直接构造模型再加载参数时，也必须严格核对缺失/多余 keys 与实际冻结的参数 SHA。

`freeze_for_world_model_pretraining(model)` 只允许以下前缀训练：
`lang_init.`, `prior.`, `posterior.`, `to_vis_embed.`, `to_reward.`, `obs_transform.`。
所有其他参数冻结；`freeze_encoders(model)` 冻结 embeddings、lang_encoder、
img_embeddings，包含 panorama 额外使用的共享 token_type_embeddings。
Glocal 的 `train()` override 会保持这些 encoder 为 eval，以避免 frozen features
仍随 dropout 改变。IR2R 时可训练非 encoder 的导航与世界模型模块。

`pretrain/model/vilmodel.py` 也有相同 backend，方便旧 joint-pretrain API 接入；
本次只训世界模型的入口由单独训练脚本控制，不能直接套原多任务 joint pretraining。

## 验证与来源

8 项独立模型测试覆盖：

1. 独立闭式 semiseparable SSD 扫描的输出与所有参数梯度一致。
2. 官方 Mamba v2.2.4 原始 step 的纯 PyTorch fallback，11 步输出和两个 cache 一致。
3. sequence、chunk、逐 step 数值一致。
4. 改变未来输入不改变过去输出。
5. full language mask/conditioning、prior 逐步与整段数值一致。
6. 三个 overshooting 分支的 batch-stack 与独立计算一致，原 cache 未改变。
7. masked instruction 或不匹配 cache 拒绝。
8. 全部共享编码器参数冻结，prior 参数仍可训练。

```bash
python -m unittest discover -s tests -p test_drama_mamba2.py -v
```

模型生产路径只依赖 torch；官方 reference 测试额外依赖 `tests/requirements-drama.txt`
中的 einops 0.8.1。测试只执行 pinned source 中的 step AST，避免加载可选 CUDA extension。
CPU 测试不替代实际 GPU/MatterSim 训练检查，也未声称已经完成完整训练或导航涨点。

来源：Drama `a50bd54c34e77d1d13e988a031733a47817098e2`；官方 Mamba v2.2.4
`95d8aba8a8c75aedcaa6143713b11e745e7cd0d9`。文件 SHA、用途见
`drama_provenance.json`，上游 Apache-2.0 license 在 `tests/vendor/MAMBA_LICENSE`。
