from datasets import load_dataset
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Trainer, TrainingArguments, Wav2Vec2Processor
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torchaudio
from functools import partial
from transformers import DataCollatorWithPadding
import matplotlib.pyplot as plt

# 定义目标采样率和音频长度
TARGET_SAMPLING_RATE = 16000
TARGET_LENGTH = 160000

# 定义全局参数
NUM_WORKERS = 22
BATCH_SIZE = 22
NUM_PROC = 22


# 自定义 DataCollator 类
class DataCollatorForWav2Vec2(DataCollatorWithPadding):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer)

    def __call__(self, features):
        input_values = [torch.tensor(f['input_values']) for f in features]
        labels = [torch.tensor(f['labels']) for f in features]
        input_values_padded = pad_sequence(input_values, batch_first=True)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        batch = {
            "input_values": input_values_padded,
            "labels": labels_padded
        }
        return batch


# 数据预处理函数
def preprocess_function(processor, examples, device):
    audio = examples["audio"]
    audio_array = audio["array"]
    original_sampling_rate = audio["sampling_rate"]

    if original_sampling_rate != TARGET_SAMPLING_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=TARGET_SAMPLING_RATE)
        audio_array = resampler(torch.tensor(audio_array).float()).numpy()

    min_val = np.min(audio_array)
    max_val = np.max(audio_array)
    if max_val != min_val:
        audio_array = (audio_array - min_val) / (max_val - min_val) * 2 - 1
    else:
        audio_array = np.zeros_like(audio_array)

    result = processor(
        audio_array,
        sampling_rate=TARGET_SAMPLING_RATE,
        return_tensors="pt",
        padding="longest"
    )

    input_values = result["input_values"].squeeze()

    if input_values.shape[0] > TARGET_LENGTH:
        input_values = input_values[:TARGET_LENGTH]
    elif input_values.shape[0] < TARGET_LENGTH:
        padding = TARGET_LENGTH - input_values.shape[0]
        input_values = torch.cat([input_values, torch.zeros(padding)], dim=0)

    encodings = processor.tokenizer(examples["sentence"],
                                    padding="longest",
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=360000) # max_length
    labels = encodings.input_ids.squeeze()

    return {
        "input_values": input_values.to(device),  # Ensure this is on GPU
        "labels": labels.to(device)  # Ensure this is on GPU
    }


# 自定义 Trainer 类
class CustomTrainer(Trainer):
    def __init__(self, *args, encoded_train_dataset=None, encoded_eval_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoded_train_dataset = encoded_train_dataset
        self.encoded_eval_dataset = encoded_eval_dataset

    def get_train_dataloader(self):
        return DataLoader(
            self.encoded_train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=self.data_collator
        )

    def get_eval_dataloader(self, eval_dataset=None):
        return DataLoader(
            self.encoded_eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=self.data_collator
        )

    def training_step(self, model, inputs):
        inputs = {key: value.to(self.args.device) for key, value in inputs.items()}
        return super().training_step(model, inputs)

    def evaluation_step(self, model, inputs):
        inputs = {key: value.to(self.args.device) for key, value in inputs.items()}
        return super().evaluation_step(model, inputs)


# 用于测试数据是否正常
def sample_test(encoded_eval_dataset, encoded_train_dataset, model, device):
    test_samples = encoded_eval_dataset.select(range(1))
    input_values = torch.tensor(test_samples["input_values"]).unsqueeze(0)
    labels = torch.tensor(test_samples["labels"]).unsqueeze(0)
    input_values = input_values.squeeze().unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_values=input_values.to(device), labels=labels.to(device))

    print(f"Loss: {outputs.loss.item()}")
    print(f"Logits: {outputs.logits}")
    print(f"Labels: {labels}")
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Logits max: {outputs.logits.max()}")
    print(f"Logits min: {outputs.logits.min()}")

    print(encoded_train_dataset.features)
    print(encoded_eval_dataset.features)

    print(encoded_train_dataset[0])  # 打印第一个训练样本
    print(encoded_eval_dataset[0])  # 打印第一个验证样本


# 绘制数据频率分布图以确定max_length
def show_datasets(dataset):
    lengths = [len(sample["audio"]["array"]) for sample in dataset["train"]]
    plt.hist(lengths, bins=50)
    plt.xlabel("Audio Length (samples)")
    plt.ylabel("Frequency")
    plt.title("Audio Length Distribution")
    plt.show()


# 数据清洗，处理掉数据集中不合理的数据
def data_cleaning(dataset):
    anomalous_indices = []
    for i, sample in enumerate(dataset["train"]):
        audio_array = sample["audio"]["array"]
        min_val = np.min(audio_array)
        max_val = np.max(audio_array)
        if max_val == min_val:
            anomalous_indices.append(i)
    print(f"Found {len(anomalous_indices)} anomalous samples.")
    print("Anomalous sample indices:", anomalous_indices)
    dataset["train"] = dataset["train"].select([i for i in range(len(dataset["train"])) if i not in anomalous_indices])
    return dataset


# 主函数
def main():
    # 自动选择训练设备，并输出
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 导入数据
    dataset = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        "zh-CN",
        trust_remote_code=True,
        cache_dir="E:/huggingface/dataset"
    )

    # 数据清洗
    dataset = data_cleaning(dataset)
    # 导入训练集和验证集
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=TARGET_SAMPLING_RATE,
        padding=True,
        return_tensors="pt"
    )
    # 导入模型
    model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")
    processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn")

    preprocess_function_with_processor = partial(preprocess_function, processor, device=device)
    # 映射数据
    encoded_train_dataset = train_dataset.map(
        preprocess_function_with_processor,
        remove_columns=["audio", "sentence"],
        num_proc=NUM_PROC
    )
    encoded_eval_dataset = eval_dataset.map(
        preprocess_function_with_processor,
        remove_columns=["audio", "sentence"],
        num_proc=NUM_PROC
    )
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=1e-6,
        max_grad_norm=0.1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    data_collator = DataCollatorForWav2Vec2(tokenizer=processor.tokenizer)
    # 训练器
    trainer = CustomTrainer(
        model=model.to(device),
        args=training_args,
        data_collator=data_collator,
        encoded_train_dataset=encoded_train_dataset,
        encoded_eval_dataset=encoded_eval_dataset
    )
    # 开始训练并输出训练结果
    trainer.train()
    results = trainer.evaluate()
    print(results)
    # 输出并保存模型
    model.save_pretrained('./fine-tuned-wav2vec2-asr')
    feature_extractor.save_pretrained('./fine-tuned-wav2vec2-asr')


if __name__ == "__main__":
    main()
