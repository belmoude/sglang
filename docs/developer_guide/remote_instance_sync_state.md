# Remote Instance 权重同步：传输 `named_parameters()` 之外的派生状态

> 配套实现见 `python/sglang/srt/model_loader/remote_sync_state.py` 及 NCCL / transfer-engine 两条 remote-instance 加载路径。

## 1. 背景与问题

`RemoteInstanceModelLoader` 允许一个新启动的实例（client）从一个已加载完成的实例（seed）直接拉取权重，而不是各自从磁盘加载。当前实现的核心流程是：

```
client: _initialize_model()                      # 仅构造网络结构（“后处理前”）
client: 从 seed 拷贝 model.named_parameters()      # NCCL broadcast / RDMA read
client: _post_load_weights(model)                # 本地重算派生状态
```

问题在于：**模型加载完成后的真实状态远不止 `named_parameters()`**。`load_weights_and_postprocess` 会产生三类“额外状态”，它们都不在 `named_parameters()` 中：

| 类别 | 例子（DeepSeek-V3 MLA） | 存在形式 |
|---|---|---|
| 派生张量 | `self_attn.w_kc` / `w_vc` / `w_scale_k` / `w_scale_v` | module 上的普通属性，**可能是非 contiguous（transpose）视图** |
| Python 标量/flag | `self_attn.w_scale`（`float` 或 `Tensor`）、`use_deep_gemm_bmm` | module 上的普通属性 |
| 张量对象属性 | `weight_scale_inv.format_ue8m0=True` | 动态 `setattr` 到某个 **named_parameter 张量对象**上 |

仅靠 client 本地重跑 `post_load_weights()` 来恢复这些状态有两个根本缺陷：

1. **顺序敏感**。正常加载顺序是 `post_load_weights()`（读原始 `kv_b_proj` 生成 `w_kc/w_vc`）→ 再 `process_weights_after_loading()`（对各量化层 requant/repack）。若某 quant method 改变了 `kv_b_proj` 的表示形式（packed/marlin/改 shape），client 端“先 process 再 post_load”会让 `post_load_weights` 读到无法识别的布局，**静默算错**。
2. **状态丢失**。`format_ue8m0` 这类挂在张量对象上的 python flag 不随 `.data.copy_()` 传输；RL/热更场景二次 `load_weights` 时会因此误判“尚未 requant”而重复处理。

**目标**：提供一套优雅、opt-in、可复用的基础设施，让 client **忠实接收** seed 的派生状态，而非本地重算，从根上消除顺序依赖与状态丢失。

## 2. 设计原则

1. **按介质分通道**。派生张量走指针/字节通道（NCCL broadcast / RDMA read），python 标量/flag 走对象通道（pickle / JSON）。二者绝不混用。
2. **声明式协议，不侵入赋值点**。模型层用 mixin 声明“同步哪些属性”，集中可读；不在每个 `bind_or_assign` 赋值处插 `register(...)`（分散、易漏、RL 二次加载会重复触发）。
3. **运行时类型路由**。同一属性运行时类型可能变化（`w_scale` 可能是 `float` 也可能是 `Tensor`），由协议方法在运行时按 `torch.is_tensor` 决定走哪个通道，名单本身不写死类型。
4. **FQN 命名空间**。用 module 的全限定名（FQN）作为 key 空间，遍历 `named_modules()` 收集，天然“按层”组织，无需手写 per-layer schema。
5. **storage 级、stride 忠实**。以底层 storage 字节为传输单位，并携带 `shape/stride/storage_offset/storage_nbytes`，正确还原非 contiguous / 带 offset / 共享 storage 的张量。
6. **向后兼容**。逐参数线协议不变；新状态以追加 payload（NCCL）或保留 key（transfer engine）携带；未接入协议的模型回退到原 `_post_load_weights` 行为。

## 3. 总体架构

```
seed (已加载完成)                              client (新启动)
─────────────────────                          ─────────────────────
collect_remote_sync_payload(model)             _initialize_model()           # 结构（attr 多为 None）
  ├─ extra_tensors: {FQN.attr -> Tensor}       拷贝 named_parameters()         # 原有通道
  ├─ manifest:      [ {name,dtype,shape,        ┌──────────────────────────────┐
  │                    stride,storage_offset,   │ ① 按 manifest 预分配 recv buf │
  │                    storage_nbytes}, ... ]   │   (RDMA 还需 register_memory) │
  └─ metadata:                                  │ ② 传输 storage 字节            │
     ├─ module_attrs: {mod_FQN: {attr: scalar}} │ ③ rebuild(as_strided) + 回填   │
     ├─ tensor_flags: {param_FQN: {flag: val}}  └──────────────────────────────┘
     └─ header: {version, uses_sync_providers}  apply_remote_sync_state(...)   # 取代 _post_load_weights
```

### 3.1 三命名空间的 metadata

非张量数据统一收敛为一份可 pickle 的 metadata，分三个 namespace：

- `module_attrs`：module 级标量，按 **module FQN** 寻址（如 `w_scale=1.0`、`use_deep_gemm_bmm=True`）。
- `tensor_flags`：张量对象 flag，按 **参数 FQN** 寻址（如 `...weight_scale_inv.format_ue8m0=True`）。
- `header`：协议版本与开关（`version`、`uses_sync_providers`）。

### 3.2 stride 忠实的张量传输

`w_kc/w_vc` 由 `transpose(1,2).contiguous().transpose(1,2)` 产生，是**非 contiguous** 视图，其 storage layout 是刻意为之（利于 bmm）。因此：

- 传输单位是**整块 untyped storage 的字节**（`storage_byte_view`），而非逻辑 `numel`；
- manifest 记录 `shape/stride/storage_offset/storage_nbytes`；
- client 收到字节后用 `Tensor.set_(storage, offset, shape, stride)` 还原（`rebuild_tensor`），布局 100% 一致；
- 该方案天然兼容 contiguous 参数（stride 即 C-order、offset=0）与共享 storage（预留 `storage_id`，见 §7）。

## 4. 关键 API（`remote_sync_state.py`）

```python
@dataclass
class RemoteSyncState:
    tensors: Dict[str, torch.Tensor]   # 局部属性名 -> 张量（字节通道）
    metadata: Dict[str, Any]           # 局部属性名 -> 标量（对象通道）

class RemoteSyncStateMixin:
    _REMOTE_SYNC_ATTRS: Tuple[str, ...] = ()      # 子类只声明这一张名单
    def get_remote_sync_state(self) -> RemoteSyncState: ...   # 运行时按类型路由
    def set_remote_sync_state(self, state) -> None: ...       # 首次 setattr / 二次 copy_

# 张量对象 flag 白名单（按 param FQN 收集/回填）
REMOTE_SYNC_TENSOR_FLAGS = ("format_ue8m0",)

# 序列化原语（stride 忠实）
storage_byte_view(t) -> uint8 view
build_manifest_entry(name, t) -> dict
rebuild_tensor(byte_buf, entry) -> Tensor

# seed 收集 / client 应用
collect_remote_sync_payload(model) -> (extra_tensors, payload)
apply_remote_sync_state(model, manifest, recv_buffers, metadata)
has_sync_providers(model) / payload_uses_sync_providers(payload)

# NCCL 传输 helper
nccl_send_remote_sync_state(model, group, device)
nccl_recv_remote_sync_state(model, group, device) -> (payload, recv_buffers)
maybe_apply_remote_sync_state(model, payload, recv_buffers) -> bool
```

模型接入只需两行：

```python
class DeepseekV2AttentionMLA(nn.Module, RemoteSyncStateMixin, ...):
    _REMOTE_SYNC_ATTRS = ("w_kc", "w_vc", "w_scale", "w_scale_k", "w_scale_v", "use_deep_gemm_bmm")
```

> `format_ue8m0` 不在 module 名单里，它通过 `collect_remote_sync_payload` 内部遍历 `state_dict(keep_vars=True)` + `REMOTE_SYNC_TENSOR_FLAGS` 自动采集。**必须用 `keep_vars=True`**，否则默认 `state_dict()` 会 detach 张量、丢掉自定义 python 属性（实现中已踩坑并修复）。

## 5. client 三阶段恢复（“每层都有”如何路由）

client 不需要知道哪些层有派生状态——seed 已把全集枚举进扁平 manifest（key 为 FQN）。client 只做：

1. **预分配**：按 manifest 为每个 `extra` 张量分配 `storage_nbytes` 的 uint8 接收缓冲（RDMA 还要 `register_memory`）。注意次序：`_initialize_model → 预分配 →(RDMA)注册 → 传输 → 回填`。
2. **传输**：NCCL 逐条 broadcast / RDMA 一次 `batch_transfer_sync_read` 填字节。
3. **回填**：`apply_remote_sync_state` 内部：
   - 用 `rpartition(".")` 把 `fqn` 拆成 `(module_fqn, attr)`，`model.get_submodule(module_fqn)` 精确定位该层；
   - `rebuild_tensor` 还原布局，按 module 分组调用 `set_remote_sync_state(tensors + module_attrs)`；
   - `apply_tensor_flags` 按 param FQN 把 `format_ue8m0` 等 setattr 回对应张量对象（同样用 `keep_vars=True`）。

dense 层 / 未接入协议的层在 manifest 中没有条目，自动跳过。

## 6. 实施步骤

### 阶段 0：核心基础设施（必做，独立可测）
1. 新增 `python/sglang/srt/model_loader/remote_sync_state.py`（§4 全部 API）。
2. 新增单测 `test/registered/unit/model_loader/test_remote_sync_state.py`：覆盖非 contiguous round-trip（stride 一致）、`format_ue8m0` 同步、`w_scale` 运行时类型路由、无 provider 回退。CPU 即可运行。

### 阶段 1：模型接入
3. `DeepseekV2AttentionMLA` 混入 `RemoteSyncStateMixin` 并声明 `_REMOTE_SYNC_ATTRS`。
4. （后续）longcat / bailing / sarvam / minicpm3 等含 `bind_or_assign` 派生张量的模型按需接入——只加 mixin + 名单，传输层零改动。

### 阶段 2：NCCL 路径接缝
5. seed `ModelRunner.send_weights_to_remote_instance`：在 `named_parameters()` broadcast 之后调用 `nccl_send_remote_sync_state(self.model, send_group, device)`。
6. client `RemoteInstanceModelLoader.load_model_from_remote_instance_by_nccl`：参数接收后调用 `nccl_recv_remote_sync_state(...)`；`maybe_apply_remote_sync_state(...)` 返回 `False`（seed 无 provider）时回退 `_post_load_weights(model)`。

### 阶段 3：transfer-engine 路径接缝
7. `remote_instance_weight_loader_utils.register_memory_region_v1/v2`：调用新增 `register_remote_sync_state(model, transfer_engine, weight_mr_dict)`，注册额外张量 storage 并把 `manifest+metadata`（含每个额外张量的 `seed_storage_ptr`）写入保留 key `__remote_sync__`。
8. client `load_model_from_remote_instance_by_transfer_engine`：从 seed weight-info 中 `pop("__remote_sync__")`；为每个额外张量分配并 `register_memory` 接收缓冲、加入 RDMA 读列表；`batch_transfer_sync_read` 后，若 `payload_uses_sync_providers` 则 `apply_remote_sync_state`，否则回退 `_post_load_weights`。

### 阶段 4：校验与回退
9. `header` 携带并校验 `version`、并行/量化配置（TP/EP rank、quant_name、deepgemm/ue8m0 开关）；不一致 **fail-fast** 而非硬同步。
10. 张量按 `(shape,dtype,stride,storage_nbytes)` 校验；缺失/多余 key 报错。

## 7. 边界、风险与未来工作

- **预分配早于注册**（RDMA）：额外张量接收缓冲必须先分配再 `register_memory`，否则无地址可注册。
- **metadata 通道**：优先 pickle（`broadcast_object_list`）；transfer-engine 经 HTTP/JSON 时，`tensor_flags` 仅含 JSON-native 标量（bool/float/int/str），张量一律走字节通道。
- **配置一致性**：seed 与 client 必须相同 model / TP / EP / quant 配置，额外张量同为本 rank 分片。
- **跳过 `process_weights_after_loading`**：client 走 sync-provider 路径时不重跑量化后处理；前提是 seed 的 param `.data` 已是后处理后格式且 shape 与 client `create_weights` 一致（DeepSeek 原生 FP8 满足）。若某 quant method 在后处理中 `register_buffer` 真实数据或改 param shape/集合，需把这些 buffer/Param 也纳入同步（它们已进 `state_dict`，由参数通道覆盖）或在 client 端补跑后处理（见下）。
- **共享 storage**：manifest 预留 `storage_id`，按 id 分配一次缓冲、多张量 `as_strided` 重建（当前 `w_kc/w_vc` 各自独立 storage，可后续实现）。
- **批量优化**：逐层小张量 broadcast 可改用 `FlattenedTensorBucket` 一次传输（参考 `_update_bucketed_weights_from_distributed`）。
- **更激进的通用方案**：对“后处理会改 param 集合/shape”的模型，可让 client 预跑 `process_weights_after_loading` 定型结构后再按 name 接收，或退回完整 `load_weights` 流程（需 seed 提供 checkpoint 原始权重）。

## 8. 测试

- **单测（CPU）**：`test_remote_sync_state.py`，已覆盖 §6 阶段 0 列项。
- **端到端（多 GPU，合并前必跑）**：起 seed + client 两实例（相同 model/TP/EP/quant），分别用 NCCL 与 transfer-engine 后端各跑一次；对拍：
  1. `state_dict()` 的 `{name: (shape, dtype, stride)}` 与“正常磁盘加载实例”全等；
  2. 相同输入下推理 logits 对齐；
  3. 用一个会在后处理中改变表示形式的量化路径（如 marlin / bf16 在线 fp8）构造回归用例，验证不再出现 §1 的静默错。
