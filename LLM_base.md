带*内容部分参考 [stanford-cs336](https://stanford-cs336.github.io/spring2025/)

<h1 id="O7mNs">BPE tokenizer*</h1>
以英文为例，初始将单个字母作为一个 token，统计最高频率的相邻 tokens，合并，重复。(BPE 可以 tokenize 不在词表中的词）

---

考虑 tokenizer 的压缩率：bytes/tokens

其他的 tokenize 方法：

1. 基于 char，缺点：有些字符（例如 emoji 表情）很罕见，词表利用率不高
2. 基于 byte，词表大小固定为 256，缺点：压缩率为 1，序列会很长
3. 基于 word，缺点：词表无上限

<h1 id="nS0Ni">内存分析*</h1>
1. float32（4byte）：pytorch 默认的数据类型，1 位符号位，8 位指数，23 位表示尾数
2. float16：1 位符号位，5 位指数，10 位表示尾数。可能会存在过小过大的问题。
3. bfloat16：1 位符号位，8 位指数，7 位表示尾数
4. fp8：1 位符号位，4 位指数，3 位表示尾数/1 位符号位，5 位指数，2 位表示尾数

```python
import torch

x = torch.rand(4, 8)

# 查看单个数值的大小(byte)
print(x.element_size()) # 4

# 查看数值个数
print(x.numel())		# 32
```

---

1. 多个 tensor 可以共享底层的存储数值，只是解读数值的方法不一样（例如view）

```python
import torch

a = torch.randn(2, 3)
b = a.view(3, 2)

# 修改a的值，b同样被修改
a[0][0] = 0.99
print(a[0][0] == b[0][0]) # true
```

怎么做到？通过 stride 确定如何看待底层数据

```python
import torch

a = torch.randn(2, 3)
print(a.stride())        # (3, 1)
print(a.stride(0))       # 3: 从a11到a21，需要跨过的元素
print(a.stride(1))       # 1: 从a11到a12，需要跨过的元素

# a11 a12 a13 a21 a22 a23
```

2. transpose/permute 会使 tensor 不连续（变成按列访问），需要使用 contiguous 在新内存中复制数据得到连续的 tensor

```python
import torch

a = torch.randn(2, 3)
b = a.transpose(1, 0)
print(b.is_contiguous()) 		# false
c = b.view(-1)           		# RuntimeError
# c = b.contiguous().view(-1)	# 正确做法
```

```python
import torch

a = torch.randn(2, 3)
b = a.transpose(1, 0)

# a,b仍然共享底层数组，但c为复制得到
c = b.contiguous().view(-1)

print(a[0][0])
print(b[0][0])
print(c[0])
b[0][0] = 0.999
print(a[0][0])
print(b[0][0])
print(c[0])
=============================
tensor(1.0906)
tensor(1.0906)
tensor(1.0906)
tensor(0.9990)
tensor(0.9990)
tensor(1.0906)
```

<h1 id="ItGJq">计算分析*</h1>
flops：浮点操作次数

flop/s：每秒的浮点操作次数

A100 性能：312 teraflop/s，即每秒 312 万亿次浮点操作

H100 性能：带稀疏性（每 4 个数里有 2 个是 0） 1979 teraflop/s，否则只有 50%

MFU：model flops utilization，实际的flop/s 除以承诺的flop/s，大于 0.5 认为是好的

:::info
矩阵乘法（前向传播）的 flops

:::

```python
B = 16384  # 数据点数量
D = 32768  # 特征维度
K = 8192   # 输出维度

x = torch.ones(B, D)
w = torch.randn(D, K)
y = x @ w

# 每个输出元素需要D次乘法，D-1次加法，近似为2D次浮点运算
# 一共有BK个输出元素
# 因此总flops为2BDK，B可以视为数据量（在llm下可认为就是token数），DK可以视为参数量（在llm下可认为就是llm参数量）
```

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1756825575333-230362f9-0ba1-4136-99f8-989f470b060f.png)

其他运算操作的 flops，相比于矩阵乘法，可以忽略

前向传播的 flops：`2 * 数据量 * 模型参数`

:::info
反向传播的 flops

:::

$ Y=XW, X\in \mathbb{R}^{B×N_{in}}, W\in \mathbb{R}^{N_{in}×N_{out}}, Y\in \mathbb{R}^{B×N_{out}} $

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1755404511682-f69a6b5b-0bdf-4507-a409-5210bf6932c9.png)

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1755404525327-83c960e7-4afd-484d-9178-9a224afcd5bb.png)

为什么输入也需要梯度？如果 model 有多层，需要传递给前一层的权重

反向传播的 flops：`4 * 数据量 * 模型参数`（需要对输入和权重都计算梯度）

:::info
构建 model

:::

1. 在使用 nn 内的标准层（如 linear ）时，不需要额外操作。但如果是自定义层，例如`self.weight = torch.randn(10, 5)`，则需要手动改为`self.weight = nn.Parameter(torch.randn(10, 5))`才能算入模型参数进行更新
2. 训练一个 fp32 参数需要的显存 = 16 = 4 + 4 + (4 + 4) bytes (parameters, gradients, adamw optimizer state)

<h1 id="tDi7A">LLM 架构*</h1>
1. pre-norm 取代 post-norm`1.`<font style="color:rgb(25, 27, 31);">规范输入数据的分布</font>`<font style="color:rgb(25, 27, 31);">2.</font>`<font style="color:rgb(25, 27, 31);">每一层的输出都直接参与残差连接，模型在训练早期阶段更容易保持梯度信号，避免梯度消失或爆炸</font>
2. rms norm 取代 layer norm，不需要减去均值，也不需要偏置参数，减少算子提速
3. 大多数 LLM 移除 mlp 的偏置，不会产生影响
4. 大多数 LLM 使用 swiglu/geglu 等门控机制的激活函数，<font style="color:rgb(25, 27, 31);">让模型动态决定信息流通或屏蔽</font>
5. 位置编码使用 rope，更依赖 token 间的相对距离
6. 超参数的设置：
    1. mlp 隐藏层维度是输入维度的 4 倍（使用 glu 类激活函数的为 2.66）
    2. 特征维度均分给多个注意力头`num_heads × head_dim = d_model`<font style="color:rgb(25, 27, 31);">头太多会令每个头维度过小、难以捕捉复杂模式；头太少则牺牲并行度</font>
    3. d<sub>model</sub>/n<sub>layer</sub> 维持在 100-200，<font style="color:rgb(25, 27, 31);">过深(n</font><sub>layer</sub><font style="color:rgb(25, 27, 31);">较大</font><sub></sub><font style="color:rgb(25, 27, 31);">)增加通信开销，推理延迟上升 (LLM 分层到不同的 GPU 上)；过宽模型虽然单层容量大，但层数偏少不利于深层特征抽象</font>
    4. 词表大小，<font style="color:rgb(25, 27, 31);">单语模型用 30K–50K，多语或超大规模系统用 100K–250K</font>
    5. <font style="color:rgb(25, 27, 31);">不使用 dropout，设置 weight_decay=0.1，dropout 会导致模型训练和预测阶段的方差出现偏移</font>

:::info
训练稳定性

:::

1. llm 训练存在 loss/gradient spike，softmax 可能会导致数值不稳定 (最后一层以及 attention 计算中)，如果输入过小，浮点数下溢，分母可能出现除 0；如果输入过大，指数运算可能发生上溢出
2. 加入 z-loss（使 softmax 分母趋于 1，为 1 时 z-loss 为 0）

$ L_Z = \lambda \cdot (\log Z)^2,\ Z = \sum_{j} \exp(z_j) $

3. 对 attention 的 Q 和 K 进行 norm，约束 softmax 的数值

<h1 id="kuJRK">MoE*</h1>
1. 在相同 flops 下参数更多，性能更好
2. 使用多个 mlp （experts）替代原来的一个 mlp，由 router 给出 topK 个 experts 与其权重，通过 experts 后进行加权求和
3. 可以使用 dense model 的 mlp 来初始化 moe model 的 experts

:::info
pytorch 实现

:::

```python
import torch
from torch import nn


# 1.使用一个mlp作为expert
# input: tokens, hidden_size
# output: tokens, hidden_size
class ExpertNetwork(nn.Module):

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.functional.relu(x)
        output = self.linear2(x)
        return output


# 2.router
# bs, seq, hidden_size->tokens, hidden_size->tokens, expert_num
class Router(nn.Module):

    def __init__(self, hidden_size, expert_num, top_k):
        super().__init__()
        self.router = nn.Linear(hidden_size, expert_num)
        self.top_k = top_k
        self.hidden_size = hidden_size

    def forward(self, x):
        # 为所有token选择expert，需要将bs,seq合并
        x = x.view(-1, self.hidden_size)
        x = self.router(x)
        x = nn.functional.softmax(x, dim=-1)
        topk_weight, topk_idx = torch.topk(x,
                                           k=self.top_k,
                                           dim=-1,
                                           sorted=False)
        # 归一化选择的k个expert的概率
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        return topk_weight, topk_idx


# 3.moe layer
class MOELayer(nn.Module):

    def __init__(self, hidden_size, intermediate_size, expert_num, top_k):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.expert_num = expert_num
        self.top_k = top_k
        self.experts = nn.ModuleList([
            ExpertNetwork(self.hidden_size, self.intermediate_size)
            for _ in range(self.expert_num)
        ])
        self.router = Router(self.hidden_size, self.expert_num, self.top_k)

    def forward(self, x):
        token_num = x.size(0) * x.size(1)
        x_flat = x.view(-1, self.hidden_size)

        # 通过路由器获得top-k专家选择的权重和索引，形状均为 (token_num, top_k)
        topk_weight, topk_idx = self.router(x_flat)

        # 初始化输出
        output = torch.zeros_like(x_flat)

        # 双循环，遍历token，遍历token选择的experts
        for token_idx in range(token_num):
            for expert_idx in range(self.top_k):
                expert = self.experts[topk_idx[token_idx, expert_idx]]
                output[token_idx] += topk_weight[token_idx, expert_idx] * expert(x_flat[token_idx])

        output = output.view(x.size(0), x.size(1), self.hidden_size)
        return output


# 4.test
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
EXPERT_NUM = 8
TOP_K = 2

inputs = torch.randn(2, 11, 4096)
moe_layer = MOELayer(HIDDEN_SIZE, INTERMEDIATE_SIZE, EXPERT_NUM, TOP_K)
outputs = moe_layer(inputs)
print(outputs.shape)
```

:::info
负载均衡

:::

1. experts 可能存在负载均衡问题，训练难度大（可能只有较少的 experts 被激活）
2. 设置负载均衡 loss，希望每个 expert 被调用的频率相同，做法见 deepseekMoE[deepseek & qwen](https://www.yuque.com/u39172896/orbyov/msl7dzgymmsk2r1w)

<h1 id="XBN8t">GPU*</h1>
:::info
CPU 与 GPU

:::

CPU：让单个任务尽快完成

GPU：任务并行

<h1 id="M7eXp">position encoding</h1>
原因：attention 没有考虑位置信息

<h2 id="lwoch">sin-cos</h2>
$ \begin{aligned}
P E_{(p o s, 2 i)} & =\sin \left(\frac{p o s}{10000^{\frac{2 i}{d_{\text {model }}}}}\right) \\
P E_{(p o s, 2 i+1)} & =\cos \left(\frac{p o s}{10000^{\frac{2 i}{d_{\text {model }}}}}\right)
\end{aligned} $

:::info
pytorch 实现

:::

```python
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)                  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)    # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *-(math.log(10000.0) / d_model)) # (d_model//2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


if __name__ == '__main__':
    x = torch.randn(1, 10, 512)
    pe = PositionalEncoding(512, 0.1)
    print(pe(x).shape)
```

$ e^{2i*(-ln10000)/d_{model}} $

+ 可以根据公式泛化；
+ 不参与训练，训练更简单；
+ $ P E_{(p o s+k)} $可以表示为$ P E_{(p o s)} $的线性变换，包含相对位置信息

<h2 id="Zsy09">RoPE</h2>
1. 包含三角函数的矩阵对 [1,0] 的变化效果等价于将其逆时针旋转 θ

$ \left [ 1,0 \right ] \begin{bmatrix} \cos \theta & \sin \theta\\  -\sin \theta & \cos \theta\end{bmatrix}=\left [ \cos \theta,\sin \theta \right ] $

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1754963273422-b3234040-eb4d-4897-b777-e71e20886967.png)

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1754963295900-131e9f55-90f0-4c25-aec7-fc467ab07b67.png)

2.上述结论也可以证明

$ \left [ r\cos \alpha ,r\sin \alpha \right ] \begin{bmatrix} \cos \theta & \sin \theta\\  -\sin \theta & \cos \theta\end{bmatrix}=[r\cos \alpha\cos \theta-r\sin \alpha\sin \theta,r\cos \alpha\sin \theta+r\sin \alpha\cos \theta]=\left [ r\cos (\alpha +\theta),r\sin(\alpha + \theta) \right ] $

3.旋转矩阵的性质

3.1 先旋转 θ<sub>1</sub>，再旋转 θ<sub>2</sub>，等价于一下旋转（θ<sub>1</sub>+ θ<sub>2</sub>）

$ R(\theta)=\begin{bmatrix} \cos \theta & \sin \theta\\  -\sin \theta & \cos \theta\end{bmatrix}
 $

$ R(\theta_1)R(\theta_2)=R(\theta_1+\theta_2)
 $

3.2 转置

$ R(\theta)^T=\begin{bmatrix} \cos \theta & -\sin \theta\\  \sin \theta & \cos \theta\end{bmatrix}=\begin{bmatrix} \cos (-\theta) & \sin (-\theta)\\  -\sin (-\theta) & \cos (-\theta)\end{bmatrix}=R(-\theta) $

4.将 rope 应用于注意力计算，q 使用 R(m)旋转，k 使用 R(n)旋转，结果包含了 m-n 的相对位置

$ q=[q_1,q_2] \quad k=[k_1,k_2]
 $

$ attention-without-rope=qk^T
 $

$ attention-with-rope=qR(m)R(n)^Tk^T=qR(m)R(-n)k^T=qR(m-n)k^T $

5.实现：可以选择任意两个 token 进行旋转，旋转角度为

$ \theta=\left ( pos\cdot \frac{1}{10000^{\frac{2i}{d}} }  \right )  $

<h1 id="vlcn0">Attention</h1>
<h2 id="rXS1o">multi-head self attention</h2>
```python
import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads, attention_dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, X, mask=None):
        # X: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = X.shape

        Q = self.query_proj(X)
        K = self.key_proj(X)
        V = self.value_proj(X)
        # QKV: (batch_size, seq_len, hidden_dim)
        # 拆分为多个头: (batch_size, num_heads, seq_len, head_dim)

        Q_state = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K_state = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V_state = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_weight=Q_state @ K_state.transpose(-2, -1)/math.sqrt(self.head_dim)
        # attention_weight: (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            attention_weight=attention_weight.masked_fill(mask==0,float('-inf'))

        attention_weight=torch.softmax(attention_weight,dim=-1)
        attention_weight = self.attention_dropout(attention_weight)
        # print(attention_weight)

        output = attention_weight @ V_state
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.output_proj(output)
        return output
```

```python
attention_mask = (torch.tensor([
    [1, 1],
    [1, 0],
    [1, 0],
]).unsqueeze(1).unsqueeze(2).repeat(1, 8, 2, 1))

x = torch.rand(3, 2, 128)
net = MultiHeadSelfAttention(128, 8)
net(x, attention_mask).shape
```

注意：

1.math.sqrt 开根号：张量相乘结果太大，softmax 趋近 1，梯度接近 0。结果服从 0 均值，1 方差，相当于归一化。

2.masked_fill：True 的位置填一个负无穷大的数

3.在attention_weight 层面进行dropout，而不是在乘 V 之后

4.mask 的 shape 与attention_value 相同

5.多头的原因：在不同语义子空间提取信息，每个头独立关注整个序列

<h2 id="igNbd">transformer decoder</h2>
```python
import math
import torch
import torch.nn as nn


# decoder layer
class DecoderLayer(nn.Module):

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.num_heads = num_heads

        # mha
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(hidden_dim, eps=1e-6)

        # ffn
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.act_fn = nn.GELU()
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(hidden_dim, eps=1e-6)

    def attention_layer(self, q, k, v, mask=None):
        # qkv:  (batch_size, num_heads, seq_len, head_dim)
        k=k.transpose(2,3)
        attention_weights = torch.matmul(q, k) / math.sqrt(self.head_dim)

        # 下三角矩阵用于mask
        if mask is not None:
            mask=mask.tril()
            attention_weights=attention_weights.masked_fill_(mask==0, float('-inf'))
        else:
            mask=torch.ones_like(attention_weights).tril()
            attention_weights=attention_weights.masked_fill_(mask==0, float('-inf'))

        attention_weights=torch.softmax(attention_weights, dim=-1)
        # print(attention_weights)
        attention_weights=self.dropout_attn(attention_weights)

        output=torch.matmul(attention_weights, v)
        # output: (batch_size, num_heads, seq_len, head_dim)

        batch_size, _, seq_len, _ = output.shape
        output=output.transpose(1,2).contiguous().view(batch_size, seq_len, -1)

        output=self.output_proj(output)
        return output


    def mha(self, X, mask=None):
        batch_size, seq_len, _ = X.shape
        q=self.q_proj(X).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k=self.k_proj(X).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v=self.v_proj(X).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        output=self.attention_layer(q, k, v, mask)

        return self.norm_attn(X+output)

    def ffn(self, X):
        up = self.up_proj(X)
        up = self.act_fn(up)
        down = self.down_proj(up)
        down = self.dropout_ffn(down)
        return self.norm_ffn(X + down)

    def forward(self, X, mask=None):
        X = self.mha(X, mask)
        X = self.ffn(X)
        return X
```

```python
x = torch.rand(3, 4, 64)
net = DecoderLayer(64, 8)
mask = (torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0],
                      [1, 1, 1,
                       0]]).unsqueeze(1).unsqueeze(2).repeat(1, 8, 4, 1))

net(x, mask).shape
```

注意：

1.流程：MHA，FFN，各自需要 Norm(X+output)

<h1 id="UMqI0">attention 变体</h1>
<h2 id="fGs6F">MQA GQA</h2>
初衷：减少 kv cache，多个token 共享 key 和 value

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1755173530297-e6233dad-55f6-48f3-8302-43f4d79e620f.png)

```python
import torch
import torch.nn as nn
import math

# 忽略了 attention_mask, attention_dropout; 
class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_key_value_head):
        super().__init__()
        assert hidden_dim % nums_head == 0 # 可以整除
        assert nums_head % nums_key_value_head == 0  # N 个 query head 为一组

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = hidden_dim // nums_head

        # 初始化 qkv o
        self.q_proj = nn.Linear(hidden_dim, nums_head * self.head_dim)  # out feature_size (nums_head * head_dim)
        # k v out shape (nums_key_value_head * head_dim)
        self.k_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)

        self.o_proj = nn.Linear(hidden_dim, hidden_dim) # input_size nums_head * head_dim

    def forward(self, X, attention_mask=None):
        # X shape (batch, seq, hidden_dim)
        batch_size, seq, _ = X.size()

        # qkv projection
        q = self.q_proj(X)  # （batch, seq, hidden_dim)
        k = self.k_proj(X)
        v = self.v_proj(X) 

        # attention_weight 目标shape 是 (batch, nums_head, seq, seq)
        q = q.view(batch_size, seq, self.nums_head, self.head_dim)
        k = k.view(batch_size, seq, self.nums_key_value_head, self.head_dim)
        v = v.view(batch_size, seq, self.nums_key_value_head, self.head_dim)

        # 关注: nums_head 和 nums_key_value_head 的关系
        q = q.transpose(1, 2) # (b, nums_head, seq, head_dim)
        k = k.transpose(1, 2) # (b, nums_key_value_head, seq, head_dim)
        v = v.transpose(1, 2)  # (b, nums_key_value_head, seq, head_dim)

        # k v repeat； （广播操作）
        k = k.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)
        v = v.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)

        attention_score = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)

        attention_weight = torch.softmax(attention_score, dim=-1)
        # （attention_mask 忽略）

        output = attention_weight @ v  # (b, nums_head, seq, head_dim)

        # output projection 变成 (b, seq, hidden_dim)
        output = output.transpose(1, 2).contiguous()
        final_output = self.o_proj(output.view(batch_size, seq, -1))

        return final_output

# 测试
x = torch.rand(3, 2, 128)
net = GroupQueryAttention(128, 8, 4)
net(x).shape
```

注：

1.nums_key_value_head=1 即为 MQA

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1756735421429-b10c9ac7-a1be-408c-9a1f-c68bc56a1e50.png)

<h2 id="NuL2i">MLA</h2>
见：[deepseek & qwen](https://www.yuque.com/u39172896/orbyov/msl7dzgymmsk2r1w)

<h1 id="Jg7YJ">BN LN RMS</h1>
:::info
为什么需要 α,β 参数

:::

不带参数，则分布被调整为零均值和单位方差，表达能力有限。通过自动学习来调整分布。

:::info
BN LN 的区别

:::

BatchNorm是对一个batch-size样本内的每个特征做归一化（图像：计算多张图片的每个通道，图像数据通常在通道维度上具有相似的统计特性），LayerNorm是对每个样本的所有特征做归一化（文本变长，计算 pad 没有意义）。

:::info
RMS norm

:::

$ \mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \cdot \gamma $

只缩放，不改变分布，保持数值稳定

<h1 id="TTv8b">FFN</h1>
Attention Layer学习序列内部，FFN 增强单个位置的特征表达。

<h1 id="sYnEp">activation function</h1>
sigmoid

relu

gelu

<font style="color:rgb(25, 27, 31);">glu</font>

swish

swiglu

<h1 id="gJrR8">BERT</h1>
区别： GPT：单向，预测下一个词。ELMo：双向，基于 RNN。

原因：有些语言理解任务，需要双向信息

任务：MLM+NSP，masked language model、next sentence prediction

方法：预训练+微调

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1752225853333-254e2312-1fb3-4cc1-9683-d06084342e83.png)

嵌入：

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1752225946495-b4e023e6-83a3-482b-87ca-923b4846a37f.png)

MLM：80% 替换为 mask，10% 替换一个 token，10% 不变

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1752226144601-48ba7548-cf30-4898-9247-ae490f5209be.png)

NSP：50%正例，50%负例

微调：句子分类，以第一个 token 做分类，或者使用对应 token 的输出。

<h1 id="yXU57">decoder only</h1>
<h2 id="n6iX3">KV cache</h2>
天气-->真好，不使用 kv cache，分两步

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1751119014120-5020e35e-4d9b-4189-915b-4145cd4c0b38.png)

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1751119051372-74e0825b-f520-488e-b3bf-d160662bd0bc.png)

使用 kv cache：<font style="color:rgb(25, 27, 31);">从第二步开始时，只需输入当前位置的 token，得到当前位置对应的 K_cur, V_cur，再拼接上一步缓存的 K_last, V_last 得到完整的 K, V，即可完成下一个 token 的预测。（下一个 token 的 logits 由最后一个 token 经过 linear 得到）</font>

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1751119151892-e55c417e-43c9-4e4e-973c-4c5fab2f2549.png)

:::info
pytorch 实现

:::

```python
import torch
import torch.nn.functional as F
from transformers import LlamaModel, LlamaConfig, LlamaForCausalLM

D = 128 # single-head-dim
V = 64  # vocab_size

class kv_cache(torch.nn.Module):
    def __init__(self, D, V):  
        super().__init__()
        self.D = D
        self.V = V
        self.Embedding = torch.nn.Embedding(V,D)
        self.Wq = torch.nn.Linear(D,D)     
        self.Wk = torch.nn.Linear(D,D)     
        self.Wv = torch.nn.Linear(D,D)
        self.lm_head = torch.nn.Linear(D,V) # LM_head
        self.cache_K = self.cache_V = None  # initial
        
    def forward(self,X):
        X = self.Embedding(X)
        Q,K,V = self.Wq(X),self.Wk(X),self.Wv(X)
        print("input_Q:", Q.shape)
        print("input_K:", K.shape)
        print("input_V:", V.shape)
        
        # Easy KV_Cache
        if self.cache_K == None:
            self.cache_K = K
            self.cache_V = V
        else:
            self.cache_K = torch.cat((self.cache_K, K), dim = 1)
            self.cache_V = torch.cat((self.cache_V, V), dim = 1)
            K = self.cache_K
            V = self.cache_V
        
        print("cache_K:", self.cache_K.shape)
        print("cache_V:", self.cache_K.shape)
        
        # ignore proj/MLP/scaled/mask/multi-head when calculate Attention
        attn =Q@K.transpose(1,2)@V
        
        # output
        output=self.lm_head(attn)
        return output

model = kv_cache(D,V)
        
# 创建数据、不使用tokenizer
X = torch.randint(0, 64, (1,10))
print(X.shape)

for i in range(4):
    print(f"\nGeneration {i} step input_shape: {X.shape}：")
    output = model.forward(X) 
    print(output.shape)
    next_token = torch.argmax(F.softmax(output, dim = -1),-1)[:,-1]
    print(next_token.shape)
    X = next_token.unsqueeze(0)
```

```python
"""
torch.Size([1, 10])

Generation 0 step input_shape: torch.Size([1, 10])：
input_Q: torch.Size([1, 10, 128])
input_K: torch.Size([1, 10, 128])
input_V: torch.Size([1, 10, 128])
cache_K: torch.Size([1, 10, 128])
cache_V: torch.Size([1, 10, 128])
torch.Size([1, 10, 64])
torch.Size([1])

Generation 1 step input_shape: torch.Size([1, 1])：
input_Q: torch.Size([1, 1, 128])
input_K: torch.Size([1, 1, 128])
input_V: torch.Size([1, 1, 128])
cache_K: torch.Size([1, 11, 128])
cache_V: torch.Size([1, 11, 128])
torch.Size([1, 1, 64])
torch.Size([1])

Generation 2 step input_shape: torch.Size([1, 1])：
input_Q: torch.Size([1, 1, 128])
input_K: torch.Size([1, 1, 128])
input_V: torch.Size([1, 1, 128])
cache_K: torch.Size([1, 12, 128])
cache_V: torch.Size([1, 12, 128])
torch.Size([1, 1, 64])
torch.Size([1])

Generation 3 step input_shape: torch.Size([1, 1])：
input_Q: torch.Size([1, 1, 128])
input_K: torch.Size([1, 1, 128])
input_V: torch.Size([1, 1, 128])
cache_K: torch.Size([1, 13, 128])
cache_V: torch.Size([1, 13, 128])
torch.Size([1, 1, 64])
torch.Size([1])
"""
```

<h2 id="w1re9">泛化 注意力退化 轨迹依赖</h2>
:::info
泛化

:::

1. 业界的实验证明decoder only结构的few shot和zero shot表现比encoder-decoder结构的表现更好；

2. 上下文学习过程中，输入的prompt可以更直接作用到decoder的每一层参数上，微调信号更强；

:::info
注意力

:::

单向注意力是满秩矩阵（softmax后下三角矩阵的对角元素都是正数）；满秩意味着有更强的表达能力，换为双向注意力机制反而会会使秩降低；

:::info
其他

:::

1.单向注意力包含隐式的位置编码

2.kv cache 更高效

3.参数更少

4.openai，scaling law

<h1 id="j3lZH"><font style="color:rgb(25, 27, 31);">梯度消失</font></h1>
在深度神经网络的反向传播过程中，梯度需要从输出层逐层传递回输入层。梯度消失是指当网络很深时，梯度在反向传播过程中变得越来越小（甚至趋近于0），导致浅层网络的权重几乎得不到有效的更新。这通常发生在激活函数的导数小于1（尤其是远小于1）的情况下，梯度在多层连乘后急剧衰减。

<h2 id="NtqBq">激活函数</h2>
传统激活函数如 Sigmoid 和 Tanh 是梯度消失的主要“元凶”之一。它们的导数在输入绝对值较大时会趋近于0（饱和区）。

relu gelu 等

<h2 id="j7tLk">归一化</h2>
在训练过程中，网络内部各层的输入分布会随着权重更新而不断变化（Internal Covariate Shift）。这可能导致某一层的输入落入激活函数（如 Sigmoid/Tanh）的饱和区，使得该层的导数非常小，进而导致反向传播到该层的梯度很小，加剧梯度消失。

BN LN 等

<h2 id="WO2Wh">残差</h2>
在非常深的网络中，即使使用了 ReLU 和 BN，梯度在多层反向传播时连乘效应仍然可能导致其逐渐衰减至很小。残差连接允许梯度直接从较深的层传递到较浅的层。 

<h2 id="RIq7x">权重初始化</h2>
不恰当的权重初始化（如过小或过大的随机值）

Kaiming Initialization

<h1 id="qjw0k">处理长文本</h1>
1.rope

2.长上下文预训练

<font style="color:rgb(25, 27, 31);">3.ALiBi：attention with linear bias</font>

<font style="color:rgb(25, 27, 31);">rope 会导致一定的计算，造成训练和推理变慢一些</font>

<font style="color:rgb(25, 27, 31);">alibi 为 attention_score 添加 bias，惩罚远距离的 token</font>

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1755086336423-9f22f279-f72c-46ce-8c0f-e42bc1ec4822.png)

<h1 id="Eq2dN">LLM 参数估计</h1>
`L`layer 数，`h`hidden dim，`V`词表大小

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1749538743534-e96c7b36-14df-4244-abfb-0a0bcdf002d9.png)

`MHA`

QKV 以及 output：(b,s,h)->(b,s,h)，需要的矩阵(h,h)，加上偏置，一共 4h<sup>2</sup>+4h

`FFN`

(b,s,h)->(b,s,4h)->(b,s,h)，需要的矩阵(h,4h)和(4h,h)，加上偏置，一共 8h<sup>2</sup>+5h

`layernorm`

与归一化的维度有关， 2h，2 次一共 4h

`input`和`output`共享：Vh

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1749539201071-687919cd-9d58-41cc-acfa-15fb0f8fa089.png)

<h1 id="V9VIS">LLM 训练显存估计</h1>
推理：float32 ，1B=4G；float16/bf16，1B=2G

训练：约为同精度推理的 4 倍

![](https://cdn.nlark.com/yuque/0/2025/png/40534419/1755308639767-d4090cc6-c531-449e-a6bc-218a948528a6.png)

<h1 id="rjQuQ">分类评估</h1>
<h2 id="qB76o">交叉熵</h2>
信息量：`I=-log<sub>2</sub>P`（例如，太阳从东升起的概率为 1，信息量为 0；概率越大，信息量越小）

信息熵：信息量的期望。如果一个系统由大量小概率事件构成，信息熵就大。

CE loss：

$ E=-[y\log(p)+(1-y)\log(1-p)] $

$ E=-\sum_{i=1}^{n} y_i\log(P_i) $

<h2 id="rMrjV">查准 precision</h2>
查准率关注的是预测为**正例**的结果中，有多少是**真正例**。  

<h2 id="ljqTw">召回 recall</h2>
召回率关注的是所有**真正例**中，有多少被模型**召回**了（预测为正）。

<h2 id="tDT9Z">F1</h2>
$ 2*precision*recall/(precision+recall) $



