
import os
# 禁用 wandb
# os.environ["WANDB_DISABLED"] = "true"

from transformers import pipeline, set_seed
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import GPT2TokenizerFast

SAVE_PATH = "./sanguo"

# test model
generator = pipeline('text-generation', model=SAVE_PATH)
set_seed(13)
txt = generator("桃园三结义", max_length=100)
print(txt)

txt = generator("曹操", max_length=100)
print(txt)

txt = generator("张飞", max_length=100)
print(txt)

txt = generator("关羽温酒斩", max_length=100)
print(txt)

txt = generator("段誉", max_length=100)
print(txt)