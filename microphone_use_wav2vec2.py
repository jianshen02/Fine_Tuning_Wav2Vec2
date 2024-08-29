import torch
import torchaudio
import pyaudio
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from concurrent.futures import ThreadPoolExecutor

# 加载微调后的模型和处理器
model = Wav2Vec2ForCTC.from_pretrained('./fine-tuned-wav2vec2-asr', local_files_only=True)
processor = Wav2Vec2Processor.from_pretrained('./fine-tuned-wav2vec2-asr', local_files_only=True)

# 确保模型在正确的设备上（CPU或GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 实时音频设置
chunk_length_sec = 1.0  # 每次处理 1.0 秒的音频
input_sample_rate = 48000  # 麦克风的采样率
model_sample_rate = 16000  # 模型的采样率
chunk = int(input_sample_rate * chunk_length_sec)  # 每次从麦克风读取的帧数
overlap = int(chunk * 0.25)  # 25%的重叠（减少重叠提高速度）
channels = 1  # 单声道
format = pyaudio.paInt16  # 16位音频格式

# 初始化pyaudio
p = pyaudio.PyAudio()

# 打开麦克风流
stream = p.open(format=format,
                channels=channels,
                rate=input_sample_rate,
                input=True,
                frames_per_buffer=chunk)

print("开始录音...")

buffer = np.zeros(0, dtype=np.float32)

# 用于异步处理的线程池
executor = ThreadPoolExecutor(max_workers=32)

def process_audio_chunk(audio_chunk):
    # 重采样到模型的采样率
    resampler = torchaudio.transforms.Resample(orig_freq=input_sample_rate, new_freq=model_sample_rate).to(device)
    audio_chunk_resampled = resampler(torch.tensor(audio_chunk).to(device))

    # 处理音频
    inputs = processor(audio_chunk_resampled.cpu().numpy(), sampling_rate=model_sample_rate, return_tensors="pt", padding=True)

    # 将数据移动到与模型相同的设备上
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # 获取模型的预测（logits）
    with torch.no_grad():
        logits = model(**inputs).logits

    # 对logits取argmax得到预测的token ID
    predicted_ids = torch.argmax(logits, dim=-1)

    # 解码ID以获得转录文本
    transcription = processor.batch_decode(predicted_ids)

    return transcription[0]

try:
    while True:
        # 从麦克风中读取数据
        data = stream.read(chunk)
        audio_input = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

        # 将新的音频数据添加到缓冲区
        buffer = np.concatenate((buffer, audio_input))

        # 当缓冲区的长度超过 chunk 长度时，进行处理
        if len(buffer) >= chunk:
            # 从缓冲区提取 chunk 长度的数据进行处理
            audio_chunk = buffer[:chunk]

            # 更新缓冲区，保留未处理的部分（overlap部分）
            buffer = buffer[chunk - overlap:]

            # 异步处理音频数据
            future = executor.submit(process_audio_chunk, audio_chunk)
            transcription = future.result()

            # 输出转录结果
            print("转录结果:", transcription)

except KeyboardInterrupt:
    print("停止录音...")
    stream.stop_stream()
    stream.close()
    p.terminate()
    executor.shutdown()
