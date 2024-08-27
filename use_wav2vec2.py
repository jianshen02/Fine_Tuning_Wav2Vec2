from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import torchaudio

# 加载微调后的模型和处理器
model = Wav2Vec2ForCTC.from_pretrained('./fine-tuned-wav2vec2-asr', local_files_only=True)
processor = Wav2Vec2Processor.from_pretrained('./fine-tuned-wav2vec2-asr', local_files_only=True)

# 确保模型在正确的设备上（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载你的音频文件
audio_input, sample_rate = torchaudio.load("./media/xiaohong.wav")

# 如果采样率不是16kHz，则进行重采样
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    audio_input = resampler(audio_input)

# 处理音频
inputs = processor(audio_input.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True)

# 将数据移动到与模型相同的设备上
inputs = {key: value.to(device) for key, value in inputs.items()}

# 获取模型的预测（logits）
with torch.no_grad():
    logits = model(**inputs).logits

# 对logits取argmax得到预测的token ID
predicted_ids = torch.argmax(logits, dim=-1)

# 解码ID以获得转录文本
transcription = processor.batch_decode(predicted_ids)

# 输出转录结果
print("转录结果:", transcription[0])
