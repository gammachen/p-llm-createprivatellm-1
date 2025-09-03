# Hello LLM

This project is the initialization training of a large language model(LLM).

## how to train
docker
```
docker build -t novel_model:beta .
# use gpu
# docker run -it --gpus all novel_model:beta sh
docker run -it novel_model:beta sh
python novel_model.py
```
linux 
```
# create env
python3 -m venv ~/.env
# active env
source ~/.env/bin/activate
# download code
git clone git@github.com:gammachen/novel_model.git
cd novel_model
# install requirements
pip install -r requirements.txt
# run train
python novel_model.py
```

## Detail

#### prepare data

First we need to prepare the data for training, we are training based on <Romeo and Juliet>.

#### train model

Tokenization is to divide the input text into meaningful subunits (tokens).
Through the following code, a new tokenizer based on our data

```python
# 导入os模块，用于操作系统相关功能，主要用于设置环境变量
import os
# 禁用 wandb（Weights & Biases）实验跟踪工具，避免训练过程中的网络连接和初始化超时问题
# 通过设置环境变量WANDB_DISABLED为"true"来完全禁用wandb功能
os.environ["WANDB_DISABLED"] = "true"

# 从Hugging Face transformers库导入核心组件
# pipeline: 用于快速创建各种NLP任务的推理管道，支持文本生成、分类等任务
# set_seed: 用于设置PyTorch、NumPy等库的随机种子，确保训练和推理结果的可重现性
from transformers import pipeline, set_seed
# GPT2Config: GPT-2模型的配置类，用于定义模型架构参数（层数、隐藏层大小、注意力头数等）
# GPT2LMHeadModel: 带有语言建模头的GPT-2模型，用于文本生成任务
# GPT2Tokenizer: GPT-2专用的分词器，用于将文本转换为模型可理解的token序列
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
# LineByLineTextDataset: 按行读取文本文件的数据集类，每行作为一个训练样本
from transformers import LineByLineTextDataset
# DataCollatorForLanguageModeling: 语言建模任务的数据整理器，负责批次数据的padding和掩码处理
from transformers import DataCollatorForLanguageModeling
# Trainer: Hugging Face提供的高级训练器，封装了完整的训练循环
# TrainingArguments: 训练参数配置类，包含学习率、批次大小、训练轮数等超参数
from transformers import Trainer, TrainingArguments
# 从tokenizers库导入自定义分词器相关组件，用于构建专门针对中文古典文学的分词器
# Tokenizer: 分词器的基础类，提供分词器的核心功能框架
from tokenizers import Tokenizer
# BPE: 字节对编码（Byte Pair Encoding）算法实现，通过统计字符对频率来学习子词分割
from tokenizers.models import BPE
# BpeTrainer: BPE算法的训练器，负责在给定语料上训练BPE模型，学习最优的词汇表
from tokenizers.trainers import BpeTrainer
# NFKC: Unicode标准化形式KC（兼容性分解后重组），用于统一不同编码的相同字符
# Sequence: 用于将多个文本处理步骤组合成序列，按顺序执行
from tokenizers.normalizers import NFKC, Sequence
# ByteLevel: 字节级预分词器，将文本转换为字节序列，能处理任何Unicode字符
from tokenizers.pre_tokenizers import ByteLevel
# ByteLevelDecoder: 字节级解码器，将token序列解码回原始文本，与ByteLevel预分词器配对使用
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
# GPT2TokenizerFast: 基于Rust实现的快速GPT-2分词器，性能优于Python版本
from transformers import GPT2TokenizerFast
# 导入wandb实验跟踪库（虽然通过环境变量被禁用，但保留导入以防后续需要）
import wandb

# 定义模型和分词器的保存路径常量
# 训练完成后，模型权重、配置文件、分词器等所有相关文件都将保存在此目录下
SAVE_PATH = "./novel_model"

# 创建基于BPE算法的自定义分词器实例
# BPE(unk_token="<unk>"): 初始化字节对编码模型，指定未知词（不在词汇表中的词）的标记为<unk>
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
# 配置文本标准化器：使用NFKC（兼容性分解后重组）统一字符编码
# 这对中文文本特别重要，因为同一个字符可能有多种Unicode编码形式
tokenizer.normalizer = Sequence([NFKC()])  # 统一字符格式，确保编码一致性
# 配置字节级预分词器：将文本转换为字节序列进行处理
# 这种方式能够处理任何Unicode字符，包括中文、标点符号和特殊字符
tokenizer.pre_tokenizer = ByteLevel()      # 字节级预分词，支持全字符集
# 配置字节级解码器：将分词后的token序列还原为可读文本
# 与ByteLevel预分词器配对使用，确保编码解码的一致性
tokenizer.decoder = ByteLevelDecoder()     # 字节级解码器，还原文本

# 定义模型训练和推理过程中需要的特殊标记列表
# <s>: 序列开始标记（Beginning of Sequence），标识文本的开始
# <pad>: 填充标记（Padding），用于批处理时将不同长度的序列对齐到相同长度
# </s>: 序列结束标记（End of Sequence），标识文本的结束
# <unk>: 未知词标记（Unknown），表示词汇表中不存在的词汇
# <mask>: 掩码标记（Mask），用于掩码语言建模任务（虽然本项目使用因果语言建模）
special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

# 创建BPE（字节对编码）训练器实例，用于在给定语料上学习最优的子词分割策略
trainer = BpeTrainer(
    # vocab_size=50000: 设置最终词汇表的大小为50000个token
    # 这个大小需要平衡模型性能和计算效率，太小会导致过多未知词，太大会增加计算开销
    vocab_size=50000, 
    # show_progress=True: 在训练过程中显示进度条，便于监控训练状态
    show_progress=True,
    # initial_alphabet: 设置初始字符集为字节级字母表（0-255的所有字节值）
    # 这确保分词器能够处理任何可能的字符，包括中文、英文、标点符号等
    initial_alphabet=ByteLevel.alphabet(),  # 字节级字符集，支持全Unicode字符
    # special_tokens: 将之前定义的特殊标记添加到词汇表中，确保它们不会被进一步分割
    special_tokens=special_tokens
)

# 定义用于训练分词器的中文古典文学语料文件列表
# "text/sanguoyanyi.txt": 三国演义全文，提供古代汉语和历史小说的语言特征
# "text/tian_long_ba_bu_all.txt": 天龙八部全文，提供武侠小说的语言特征和现代中文表达
files = ["text/sanguoyanyi.txt", "text/tian_long_ba_bu_all.txt"]
# 在指定的文本文件上训练BPE分词器
# 训练过程会统计字符对的出现频率，学习最优的子词分割策略，构建适合中文文本的词汇表
tokenizer.train(files, trainer)

# 将自定义训练的分词器包装为GPT2TokenizerFast格式
# GPT2TokenizerFast是基于Rust实现的高性能分词器，与Transformers库完全兼容
# tokenizer_object参数指定使用之前训练好的自定义分词器作为底层实现
newtokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
# 将包装后的分词器保存到指定目录
# 保存内容包括：词汇表文件、分词器配置、特殊标记映射等
# 后续可以直接从此路径加载分词器，无需重新训练
newtokenizer.save_pretrained(SAVE_PATH)  # 保存到./sanguo目录

# 从本地保存路径重新加载GPT-2分词器
# 使用GPT2Tokenizer.from_pretrained()方法加载之前保存的自定义分词器
tokenizer = GPT2Tokenizer.from_pretrained(SAVE_PATH)
# 显式添加特殊标记到分词器中，确保模型能正确识别和处理这些特殊情况
# eos_token="</s>": 序列结束标记，用于标识文本生成的结束点
# bos_token="<s>": 序列开始标记，用于标识文本生成的起始点
# unk_token="<unk>": 未知词标记，用于处理词汇表外的词汇
# pad_token="<pad>": 填充标记，用于批处理时对齐不同长度的序列
# mask_token="<mask>": 掩码标记，用于掩码语言建模（本项目中备用）
tokenizer.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>",
                             "unk_token": "<unk>", "pad_token": "<pad>", "mask_token": "<mask>"})

# 创建GPT-2模型的配置对象，定义模型的架构参数
config = GPT2Config(
    # vocab_size: 设置模型词汇表大小，必须与分词器的词汇表大小保持一致
    # 这决定了模型输出层的维度和嵌入层的大小
    vocab_size=tokenizer.vocab_size,
    # bos_token_id: 设置序列开始标记的数字ID，用于文本生成时的起始标记
    bos_token_id=tokenizer.bos_token_id,
    # eos_token_id: 设置序列结束标记的数字ID，用于判断文本生成的结束条件
    eos_token_id=tokenizer.eos_token_id
    # 其他参数使用GPT2Config的默认值：
    # - n_positions=1024 (最大序列长度)
    # - n_embd=768 (嵌入维度)
    # - n_layer=12 (Transformer层数)
    # - n_head=12 (注意力头数)
)

# 根据配置创建GPT-2语言模型实例
# GPT2LMHeadModel包含完整的GPT-2架构：嵌入层、多层Transformer块、语言建模头
# 模型权重随机初始化，需要通过训练来学习语言模式
model = GPT2LMHeadModel(config)

# 创建第一个训练数据集：三国演义文本数据集
# LineByLineTextDataset会按行读取文本文件，每行作为一个独立的训练样本
dataset = LineByLineTextDataset(
    # tokenizer: 指定使用之前配置的自定义分词器进行文本tokenization
    tokenizer=tokenizer,
    # file_path: 指定三国演义文本文件的路径，包含完整的古典小说内容
    file_path="./text/sanguoyanyi.txt",
    # block_size=128: 设置每个训练样本的最大token长度为128
    # 相比原来的32，128能更好地捕捉中文古典小说的语言模式和上下文信息
    # 这个长度适合中文的句子结构，能包含完整的语义单元
    block_size=128,
)

# 创建第二个训练数据集：天龙八部文本数据集
# 添加第二个数据集可以增加训练语料的多样性，提高模型的泛化能力
dataset2 = LineByLineTextDataset(
    # tokenizer: 使用相同的分词器确保两个数据集的token化方式一致
    tokenizer=tokenizer,
    # file_path: 指定天龙八部文本文件路径，提供武侠小说的语言风格
    file_path="./text/tian_long_ba_bu_all.txt",
    # block_size=128: 与第一个数据集保持相同的序列长度，确保训练的一致性
    # 128个token足以包含天龙八部中的完整对话和描述片段
    block_size=128,
)

# 合并两个数据集，创建包含多样化中文文学内容的综合训练数据集
# 通过+操作符将两个LineByLineTextDataset对象合并，增加训练数据的丰富性
# 合并后的数据集包含古典历史小说和现代武侠小说的语言特征
dataset = dataset + dataset2

# 创建语言建模任务的数据整理器（Data Collator）
# 负责将单个样本组织成训练批次，处理padding、attention mask等
data_collator = DataCollatorForLanguageModeling(
    # tokenizer: 指定用于处理文本的分词器，用于padding和特殊标记处理
    tokenizer=tokenizer, 
    # mlm=False: 设置为False表示使用因果语言建模（Causal Language Modeling）
    # 因果语言建模是GPT系列模型的标准训练方式，模型只能看到当前位置之前的token
    mlm=False, 
    # mlm_probability=0.15: 掩码语言建模的掩码概率（在CLM模式下不生效）
    # 保留此参数是为了代码的完整性，实际训练中不会使用
    mlm_probability=0.15
)

# 配置模型训练的超参数和训练策略
# TrainingArguments包含了训练过程中的所有重要参数设置
training_args = TrainingArguments(
    # output_dir: 指定训练输出目录，用于保存模型检查点、日志和其他训练产物
    output_dir="./output",
    # overwrite_output_dir=True: 允许覆盖已存在的输出目录，避免路径冲突错误
    overwrite_output_dir=True,
    # num_train_epochs=20: 设置完整的训练轮数为20个epoch
    # 每个epoch表示模型完整遍历一次所有训练数据
    num_train_epochs=20,
    # per_gpu_train_batch_size=16: 设置每个GPU的批次大小为16个样本
    # 实际的全局批次大小 = per_gpu_train_batch_size × GPU数量
    per_gpu_train_batch_size=16,
    # save_steps=2000: 每训练2000步保存一次模型检查点
    # 这样可以在训练中断时从最近的检查点恢复训练
    save_steps=2000,
    # save_total_limit=2: 最多保留2个检查点文件，自动删除较旧的检查点
    # 这有助于节省磁盘空间，避免检查点文件过多
    save_total_limit=2,
    # 其他参数使用默认值：learning_rate=5e-5, weight_decay=0.0, warmup_steps=0等
)

# 创建Hugging Face Trainer实例，整合所有训练组件
# Trainer封装了完整的训练循环，包括前向传播、反向传播、优化器更新等
trainer = Trainer(
    # model: 指定要训练的GPT-2模型实例，包含随机初始化的权重
    model=model,
    # args: 传入训练参数配置，控制训练过程的各个方面
    args=training_args,
    # data_collator: 指定数据整理器，负责批次数据的组织和预处理
    data_collator=data_collator,
    # train_dataset: 指定训练数据集，包含合并后的三国演义和天龙八部文本
    train_dataset=dataset,
    # 注意：这里没有设置eval_dataset，表示只进行训练而不进行验证
)

# 启动模型训练过程
# trainer.train()会执行完整的训练循环，包括：
# 1. 数据加载和批次处理
# 2. 前向传播计算损失
# 3. 反向传播计算梯度
# 4. 优化器更新模型权重
# 5. 定期保存检查点
trainer.train()
# 训练完成后，将最终的模型权重和配置保存到指定目录
# 保存内容包括：模型权重文件、配置文件、生成配置等
# 这样可以在后续直接加载训练好的模型进行推理
model.save_pretrained(SAVE_PATH)

# 测试阶段：验证训练好的模型的文本生成能力
# 创建文本生成管道（pipeline），自动加载保存的模型和分词器
# pipeline会处理文本预处理、模型推理、后处理等完整流程
generator = pipeline('text-generation', model=SAVE_PATH)
# 设置随机种子为13，确保每次运行生成相同的结果，便于调试和比较
set_seed(13)
# 使用"吕布"作为起始提示词（prompt）进行文本生成
# max_length=30: 设置生成文本的最大长度为30个token（包括输入的提示词）
# 选择"吕布"是因为这是三国演义中的重要人物，能测试模型对古典文学的理解
txt = generator("吕布", max_length=30)
# 打印生成的文本结果，查看模型是否学会了中文古典文学的语言风格
print(txt)
```

tree text:
```
text
├── romeo_and_juliet.txt
├── sanguoyanyi.txt
└── tian_long_ba_bu_all.txt
```

ls ./novel_model and find added files.
```
[ 352]  novel_model
├── [  29]  added_tokens.json
├── [ 743]  config.json
├── [ 111]  generation_config.json
├── [882K]  merges.txt
├── [474M]  model.safetensors
├── [  99]  special_tokens_map.json
├── [4.2M]  tokenizer.json
├── [1.3K]  tokenizer_config.json
└── [1.2M]  vocab.json
```

#### output

```python
[ 192]  output
├── [ 480]  checkpoint-4000
│   ├── [  28]  added_tokens.json
│   ├── [ 742]  config.json
│   ├── [ 111]  generation_config.json
│   ├── [ 56K]  merges.txt
│   ├── [347M]  model.safetensors
│   ├── [694M]  optimizer.pt
│   ├── [ 14K]  rng_state.pth
│   ├── [1.0K]  scheduler.pt
│   ├── [ 692]  special_tokens_map.json
│   ├── [1.4K]  tokenizer_config.json
│   ├── [2.1K]  trainer_state.json
│   ├── [5.2K]  training_args.bin
│   └── [121K]  vocab.json
├── [ 480]  checkpoint-5200
│   ├── [  28]  added_tokens.json
│   ├── [ 742]  config.json
│   ├── [ 111]  generation_config.json
│   ├── [ 56K]  merges.txt
│   ├── [347M]  model.safetensors
│   ├── [694M]  optimizer.pt
│   ├── [ 14K]  rng_state.pth
│   ├── [1.0K]  scheduler.pt
│   ├── [ 692]  special_tokens_map.json
│   ├── [1.4K]  tokenizer_config.json
│   ├── [2.4K]  trainer_state.json
│   ├── [5.2K]  training_args.bin
│   └── [121K]  vocab.json
└── [ 416]  runs
    ├── [  96]  Aug15_20-39-21_shhaofudeMacBook-Pro.local
    │   └── [4.9K]  events.out.tfevents.1755261563.shhaofudeMacBook-Pro.local.29168.0
    ├── [  96]  Aug17_10-59-23_shhaofudeMacBook-Pro.local
    │   └── [4.9K]  events.out.tfevents.1755399566.shhaofudeMacBook-Pro.local.93327.0
    ├── [  96]  Aug17_11-14-28_shhaofudeMacBook-Pro.local
    │   └── [6.6K]  events.out.tfevents.1755400469.shhaofudeMacBook-Pro.local.2188.0
    ├── [  96]  Aug17_14-15-24_shhaofudeMacBook-Pro.local
    │   └── [4.1K]  events.out.tfevents.1755411326.shhaofudeMacBook-Pro.local.33612.0
    ├── [  96]  Aug17_14-17-50_shhaofudeMacBook-Pro.local
    │   └── [4.9K]  events.out.tfevents.1755411472.shhaofudeMacBook-Pro.local.35000.0
    ├── [  96]  Aug17_14-22-06_shhaofudeMacBook-Pro.local
    │   └── [4.1K]  events.out.tfevents.1755411729.shhaofudeMacBook-Pro.local.37230.0
    ├── [  96]  Aug17_14-32-59_shhaofudeMacBook-Pro.local
    │   └── [4.9K]  events.out.tfevents.1755412382.shhaofudeMacBook-Pro.local.42888.0
    ├── [  96]  Aug17_14-44-01_shhaofudeMacBook-Pro.local
    │   └── [4.1K]  events.out.tfevents.1755413044.shhaofudeMacBook-Pro.local.48300.0
    ├── [  96]  Aug17_14-44-55_shhaofudeMacBook-Pro.local
    │   └── [4.1K]  events.out.tfevents.1755413097.shhaofudeMacBook-Pro.local.48987.0
    ├── [  96]  Aug17_14-47-19_shhaofudeMacBook-Pro.local
    │   └── [4.9K]  events.out.tfevents.1755413241.shhaofudeMacBook-Pro.local.51220.0
    └── [  96]  Aug17_15-05-29_shhaofudeMacBook-Pro.local
        └── [7.3K]  events.out.tfevents.1755414331.shhaofudeMacBook-Pro.local.58444.0

15 directories, 37 files
```

#### 使用方法

#### 1. 训练模型
```bash
# 训练并生成示例
python train_small_model.py

# 仅训练模型
python train_small_model.py --mode train

# 训练并保存到自定义路径
python train_small_model.py --mode train --model_path ./my_model
```

#### 2. 生成文本

**使用训练脚本生成：**
```bash
# 使用已训练的模型生成文本
python train_small_model.py --mode generate

# 指定提示词生成
python train_small_model.py --mode generate --prompt "郭靖和黄蓉"

# 指定生成长度
python train_small_model.py --mode generate --prompt "乔峰" --max_length 200
```

**使用独立生成脚本：**
```bash
# 交互式生成
python generate_text.py --interactive

# 指定提示词生成
python generate_text.py --prompt "吕布手持方天画戟" --max_length 150

# 批量生成
python generate_text.py --batch "吕布" "乔峰" "郭靖"

# 指定模型路径
python generate_text.py --model_path ./my_model --prompt "诸葛亮"
```

#### 3. 测试模型

```python
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='./novel_model')
set_seed(42)
txt = generator("Hello", max_length=30)
print(txt)
```
