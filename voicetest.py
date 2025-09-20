import os
import math
from pydub import AudioSegment
import logging

# 设置日志，方便查看输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 定义分析参数 ---
ANALYSIS_DURATION_MS = 50 # 分析结尾多少毫秒的音频

def analyze_audio_tail(file_path: str):
    """
    分析 WAV 文件最后 ANALYSIS_DURATION_MS 毫秒的平均能量 (dBFS)。
    """
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return None, None

    try:
        audio = AudioSegment.from_wav(file_path)
        audio_duration_ms = len(audio) # 音频总时长

        if audio_duration_ms < ANALYSIS_DURATION_MS:
            logger.warning(f"音频 '{file_path}' (总时长 {audio_duration_ms}ms) 短于分析时长 {ANALYSIS_DURATION_MS}ms。")
            # 对于极短音频，直接返回其平均 dBFS
            return audio.dBFS, audio_duration_ms
        
        # 提取音频结尾片段
        tail_segment = audio[-ANALYSIS_DURATION_MS:]
        tail_dbfs = tail_segment.dBFS

        logger.info(f"文件: '{file_path}'")
        logger.info(f"  总时长: {audio_duration_ms}ms")
        logger.info(f"  最后 {ANALYSIS_DURATION_MS}ms 的平均能量: {tail_dbfs:.2f} dBFS")
        
        return tail_dbfs, audio_duration_ms

    except Exception as e:
        logger.error(f"分析音频 '{file_path}' 时发生错误: {e}", exc_info=True)
        return None, None

# --- 在这里指定您的WAV文件路径 ---
if __name__ == "__main__":
    # 请替换为您的实际文件路径
    
    # 示例：一个正常结尾的音频
    normal_audio_path = "F:\\software\\AudioBookServerV2\\output\\欢想世界\\wavs\\第446章受国之垢\\0036-郭煌-老头7.wav" 
    # 示例：一个戛然而止的音频
    abrupt_audio_path = "F:\\software\\AudioBookServerV2\\output\\欢想世界\\wavs\\第446章受国之垢\\0038-郭煌-老头7.wav"

    logger.info("\n--- 分析正常结尾音频 ---")
    analyze_audio_tail(normal_audio_path)

    logger.info("\n--- 分析戛然而止音频 ---")
    analyze_audio_tail(abrupt_audio_path)

    logger.info("\n--- 尝试分析更多文件以校准参数 ---")
    # 可以添加更多文件进行测试
    # analyze_audio_tail("path/to/another/normal.wav")
    # analyze_audio_tail("path/to/another/abrupt.wav")