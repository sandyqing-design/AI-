import os
import sys
import shutil
import argparse
import base64
import logging
import random
import io
import json
import re
import time
import requests
from typing import Optional, List, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
import soundfile as sf
from pydub.effects import normalize, low_pass_filter
from pydub.scipy_effects import high_pass_filter
import zipfile 
from fastapi.responses import StreamingResponse
import urllib.parse
import string
import asyncio
import math
from pydub.silence import detect_leading_silence
import uuid
from pydub.effects import normalize as pydub_normalize
import uvicorn


# --- 基本配置和目录定义 ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECTS_DIR = "projects"
WAV_DIR = "wav"
OUTPUT_DIR = "output"
TEMP_DIR = "temp_prompts"
for dir_path in [PROJECTS_DIR, WAV_DIR, OUTPUT_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

app = FastAPI(title="AI Voice Studio Pro - Backend Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


CATEGORIES_FILE = os.path.join(WAV_DIR, 'timbre_categories.json')

TTS_TAIL_ANALYSIS_DURATION_MS = 30  # 分析结尾多少毫秒的音频
TTS_TAIL_ENERGY_THRESHOLD_DBFS = -40  # 能量阈值，高于此值则判定为可能截断
TTS_GENERATION_MAX_RETRIES = 5  # 后端 TTS 生成的内部最大重试次数

# =================================================================
#               API MODELS (Pydantic)
# =================================================================
class MergeCharactersRequest(BaseModel):
    novel_name: str
    target_name: str
    source_names: List[str]
    chapter_files: List[str] 
    
class ChapterTxt(BaseModel):
    id: int
    title: str
    content: str

class ProcessTxtRequest(BaseModel):
    novel_name: str
    chapter_titles: List[str]

class TTSRequestV2(BaseModel):
    novel_name: str; chapter_name: str; row_index: int; speaker: str; timbre: str;
    tts_text: str; prompt_audio: str; prompt_text: str;
    inference_mode: str; instruct_text: Optional[str] = None
    tts_model: Optional[str] = None # 新增字段

class SpliceRequest(BaseModel):
    novel_name: str; chapter_name: str; wav_files: List[str]

class CharactersInChaptersRequest(BaseModel):
    novel_name: str; chapter_files: List[str]

class UpdateConfigRequest(BaseModel):
    novel_name: str; config_data: dict

class UpdateChapterRequest(BaseModel):
    filepath: str
    content: List[Dict]
    
class SearchSentencesRequest(BaseModel):
    novel_name: str
    character_name: str
    chapter_titles: List[str]
    
class EffectRequest(BaseModel):
    novel_name: str
    chapter_name: str
    file_name: str
    effect_type: str
    
class DownloadRequest(BaseModel):
    file_paths: List[str]    
    
class ProcessSingleChapterRequest(BaseModel):
    novel_name: str
    chapter_title: str    
    model_name: Optional[str] = None # model_name 在预览时不是必需的
    force_regenerate: Optional[bool] = False
    preview_only: Optional[bool] = False # 新增字段

class ChoralRequest(BaseModel):
    novel_name: str
    chapter_name: str
    row_index: int
    tts_text: str
    selected_timbres: List[str]  
    original_speaker: str  # 原始音频的角色名
    original_timbre: str   # 原始音频的音色名  
    tts_model: Optional[str] = None # <-- 新增
    
class DeepAnalyzeRequest(BaseModel):
    novel_name: str
    character_name: str
    model_name: str
    
class STTResponse(BaseModel):
    status: str
    text: Optional[str] = None
    message: Optional[str] = None
    
class CreateCategoryRequest(BaseModel):
    category_name: str

class SetTimbreCategoryRequest(BaseModel):
    timbre_name: str
    category_name: str

class ReplaceRule(BaseModel):
    original_word: str
    replacement_word: str
    description: Optional[str] = None

class UpdateReplaceDictRequest(BaseModel):
    rules: List[ReplaceRule]
    
# =================================================================
#               HELPER FUNCTIONS
# =================================================================
PROMPT_TEMPLATE = """
请将我提供的小说文本转换为有声书JSON格式。严格按照以下要求进行转换：
1. 输出格式必须是有效的JSON数组，每个对话或旁白为一个对象
2. 每个对象必须包含以下字段：
   - speaker: 说话者姓名（"旁白"或者角色名）
   - content: 对话或旁白内容，删除内容中不成对的引号
   - tone: 语气描述（包括“正常”、“愤怒”、“开心”、“伤心”等情绪的描述）
   - intensity: 语气强度，范围1-10的整数值
   - delay: 与上一句话之间的停顿时间（毫秒）
3. 旁白部分：
   - speaker设置为"旁白"
   - tone通常设置为"正常"
   - intensity通常设置为5（中等强度）
   - delay根据内容长度和情境设置，通常在300-800毫秒之间
4. 角色对话部分：
   - speaker设置为角色名称，需要判断说话者的角色名字，对于同一个人说的话，speaker设置为同一个角色名称，使用最合适的角色名字
   - tone需要根据对话内容和情境描述具体情感（如"愤怒"、"惊讶"、"低声念叨"等）
   - intensity根据情感强度，设置范围1-10，感情越强烈值越大
   - delay根据对话节奏设置，通常在100-1500毫秒之间
5. 长段落和长对话处理规则：
   - 超过100个字的段落应拆分，每个拆分后的片段尽量保持在50个字左右
   - 拆分后保持相同的speaker、tone和intensity
   - 选择在自然停顿处拆分，例如句号、逗号、省略号
6. 特殊要求：
   - 所有引号必须正确转义
   - 不要添加额外的字段
   - 保持原文的语意完整性
   - 不要输出content中只含有标点符号的内容
   - 不能改变文章词句的前后顺序
   - 角色说的对话内容必须保持原样，不要修改或遗漏，长句子可以分开输出
   - 绝对不能随意去除旁白的内容，只去除无意义的旁白内容，例如旁白内容中只有角色名

请严格遵循以上所有规则。你的回复必须是且仅是一个符合上述规范的、不包含任何额外解释性文字或Markdown标记的JSON数组。
以下是需要转换的小说章节内容："""
MAX_RETRIES = 3
MODEL_NAME = "gemini-2.5-flash"
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
CONFIG_FILE = "config.json"
def initialize_llm_config():
    default_config = {
        "general": {
            "default_model": "gemini",
            "proxy": {
                "enabled": False,
                "protocol": "socks5h",
                "address": "127.0.0.1",
                "port": "1080"
            },
            "default_tts_model": "cosyvoice_v2"
        },
        "audio_export": {
            "format": "mp3",
            "quality": "256k"
        },
        "tts_models": { # 新增
            "cosyvoice_v2": {
                "display_name": "CosyVoice2",
                "endpoint": "http://127.0.0.1:5010/api/tts"
            },
            "indextts_v1.5": {
                "display_name": "IndexTTS V1.5",
                "endpoint": "http://127.0.0.1:5020/api/tts"
            }
        },
        "models": {
            "gemini": {
                "display_name": "Gemini",
                "model_name": "gemini-2.5-flash",
                "api_key": "",
                "max_chars": 8000,
                "use_proxy": True
            },
            "aliyun": {
                "display_name": "阿里云平台",
                "model_name": "deepseek-r1",
                "api_key": "",
                "max_chars": 6000,
                "use_proxy": False
            }
        },
        "elevenlabs": {
            "api_key": ""
        }
    }
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=4)
        logger.info(f"'{CONFIG_FILE}' 未找到，已创建默认配置文件。")

@app.get("/api/get_llm_config")
async def get_llm_config():
    if not os.path.exists(CONFIG_FILE):
        raise HTTPException(status_code=404, detail="LLM配置文件未找到。")
    return FileResponse(CONFIG_FILE)

class LLMConfigRequest(BaseModel):
    config: Dict

@app.post("/api/update_llm_config")
async def update_llm_config(req: LLMConfigRequest):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(req.config, f, ensure_ascii=False, indent=4)
        return {"status": "success", "message": "模型配置已成功保存。"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"写入配置文件失败: {e}")
        
async def generate_with_qwen(chapter_content: str, model_id: str, api_key: str, max_chars: int, proxies: Optional[Dict]) -> List[Dict]:
    text_chunks = smart_chunk_text(chapter_content, max_length=max_chars)
    all_json_parts = []
    
    async def process_chunk(chunk_text, index):
        """Inner function to process a single chunk."""
        chunk_prompt = PROMPT_TEMPLATE + '\n\n' + chunk_text
        api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_id,
            "input": {"messages": [{"role": "system", "content": "You are a helpful assistant that strictly follows user instructions."}, {"role": "user", "content": chunk_prompt}]},
            "parameters": {"result_format": "message"}
        }
        
        for attempt in range(MAX_RETRIES):
            logger.info(f"    - (Qwen API Call) Processing chunk {index + 1}/{len(text_chunks)}, Attempt {attempt + 1}...")
            try:
                response = requests.post(api_url, headers=headers, json=payload, proxies=proxies, timeout=300)
                response.raise_for_status()
                response_data = response.json()
                
                if response_data.get("output", {}).get("choices"):
                    message = response_data["output"]["choices"][0].get("message", {})
                    api_text = message.get("content")
                    if api_text:
                        if api_text.strip().startswith("```json"):
                            api_text = api_text.strip()[7:-3].strip()
                        parsed_json = validate_and_parse_json_array(api_text)
                        if parsed_json is not None:
                            return parsed_json
                raise ValueError(f"API响应无效或内容格式不正确: {json.dumps(response_data)}")
            except Exception as e:
                logger.warning(f"    - (Qwen API Call) Chunk {index + 1} Attempt {attempt + 1} Failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(5)
                else:
                    return None
    
    for i, chunk in enumerate(text_chunks):
        json_part = await process_chunk(chunk, i)
        if json_part:
            all_json_parts.extend(json_part)
        else:
            logger.error(f"Chunk {i+1} failed to process after all retries. Skipping this chunk.")
            
    if not all_json_parts and text_chunks:
        raise Exception("所有文本块都未能成功处理。")
        
    logger.info(f"所有块处理完毕，合并了 {len(all_json_parts)} 条JSON记录。")
    return all_json_parts

async def generate_with_gemini(chapter_content: str, model_id: str, api_key: str, max_chars: int, proxies: Optional[Dict]) -> List[Dict]:
    text_chunks = smart_chunk_text(chapter_content, max_length=max_chars)
    all_json_parts = []

    async def process_chunk(chunk_text, index):
        chunk_prompt = PROMPT_TEMPLATE + '\n\n' + chunk_text
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
        headers = {'Content-Type': 'application/json', 'x-goog-api-key': api_key}
        payload = {"contents": [{"parts": [{"text": chunk_prompt}]}], "safetySettings": SAFETY_SETTINGS, "generationConfig": {"response_mime_type": "application/json"}}

        for attempt in range(MAX_RETRIES):
            logger.info(f"    - (Gemini API Call) Processing chunk {index + 1}/{len(text_chunks)}, Attempt {attempt + 1}...")
            try:
                response = requests.post(api_url, headers=headers, json=payload, proxies=proxies, timeout=300)
                response.raise_for_status()
                response_data = response.json()
                if not response_data.get('candidates'): raise ValueError(f"API响应无效: {json.dumps(response_data)}")
                candidate = response_data['candidates'][0]
                finish_reason = candidate.get('finishReason')
                if finish_reason != "STOP": raise ValueError(f"模型生成异常，原因: {finish_reason}。")
                api_text = candidate.get('content', {}).get('parts', [{}])[0].get('text')
                if api_text is None: raise ValueError("API响应中 'content.parts.text' 字段缺失。")
                parsed_json = validate_and_parse_json_array(api_text)
                if parsed_json is not None:
                    logger.info(f"    - (Gemini API Call) Chunk {index + 1} Succeeded.")
                    return parsed_json
                else:
                    raise ValueError(f"返回内容不是一个完整的JSON数组。")
            except Exception as e:
                logger.warning(f"    - (Gemini API Call) Chunk {index + 1} Attempt {attempt + 1} Failed: {e}")
                if attempt < MAX_RETRIES - 1: await asyncio.sleep(5)
        return None

    tasks = [process_chunk(chunk, i) for i, chunk in enumerate(text_chunks)]
    results = await asyncio.gather(*tasks)
    for result in results:
        if result: all_json_parts.extend(result)

    if not all_json_parts and text_chunks: raise Exception("所有文本块都未能成功处理。")
    return all_json_parts    
    
@app.post("/api/merge_characters")
async def merge_characters(req: MergeCharactersRequest):
    """
    Merges source character names into a target name.
    If the timbre/voice for the source and target characters are the same,
    it will intelligently rename the corresponding WAV files instead of requiring regeneration.
    """
    novel_name = req.novel_name
    target_name = req.target_name
    source_names = req.source_names
    chapter_files = req.chapter_files

    if not all([novel_name, target_name, source_names]):
        raise HTTPException(status_code=400, detail="请求参数不完整。")
    
    if target_name in source_names:
        raise HTTPException(status_code=400, detail="目标名称不能包含在源名称列表中。")

    project_dir = os.path.join(PROJECTS_DIR, novel_name)
    json_dir = os.path.join(project_dir, 'chapters_json')
    profiles_path = os.path.join(project_dir, 'character_profiles.json')
    timbres_path = os.path.join(project_dir, 'character_timbres.json')
    output_wav_base_dir = os.path.join(OUTPUT_DIR, novel_name, 'wavs')

    if not os.path.isdir(json_dir):
        return {"status": "success", "message": "没有找到可处理的章节文件。"}

    # --- 1. 加载音色配置 ---
    character_timbres = {}
    if os.path.exists(timbres_path):
        with open(timbres_path, 'r', encoding='utf-8') as f:
            character_timbres = json.load(f)
    
    target_timbre = character_timbres.get(target_name)
    source_names_set = set(source_names)
    
    modified_files_count = 0
    renamed_wav_count = 0

    try:
        # --- 2. 遍历并修改所有章节JSON文件 ---
        for chapter_filename in chapter_files:
            if not chapter_filename.endswith('.json'):
                continue

            chapter_json_path = os.path.join(json_dir, chapter_filename)
            file_modified = False
            
            with open(chapter_json_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            safe_chapter_name = os.path.splitext(chapter_filename)[0]
            chapter_wav_dir = os.path.join(output_wav_base_dir, safe_chapter_name)

            for index, item in enumerate(content):
                current_speaker = item.get('speaker')
                if current_speaker in source_names_set:
                    # --- 3. 智能判断与重命名WAV文件 ---
                    source_timbre = character_timbres.get(current_speaker)
                    
                    # 核心条件: 仅当源音色存在，且与目标音色相同时，才进行重命名
                    if source_timbre and source_timbre == target_timbre:
                        # 构建旧文件名和新文件名
                        safe_speaker_old = "".join(c for c in current_speaker if c.isalnum() or c in " _-").rstrip()
                        safe_speaker_new = "".join(c for c in target_name if c.isalnum() or c in " _-").rstrip()
                        safe_timbre_name = "".join(c for c in source_timbre if c.isalnum() or c in " _-").rstrip()

                        old_wav_name = f"{index:04d}-{safe_speaker_old}-{safe_timbre_name}.wav"
                        new_wav_name = f"{index:04d}-{safe_speaker_new}-{safe_timbre_name}.wav"
                        
                        old_wav_path = os.path.join(chapter_wav_dir, old_wav_name)
                        new_wav_path = os.path.join(chapter_wav_dir, new_wav_name)

                        # 执行重命名
                        if os.path.exists(old_wav_path):
                            try:
                                os.rename(old_wav_path, new_wav_path)
                                renamed_wav_count += 1
                                logger.info(f"WAV重命名: {old_wav_path} -> {new_wav_path}")
                            except OSError as e:
                                logger.error(f"重命名WAV文件失败: {e}")

                    # --- 4. 修改JSON内容 ---
                    item['speaker'] = target_name
                    file_modified = True
            
            if file_modified:
                with open(chapter_json_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
                modified_files_count += 1
        
        logger.info(f"在 {modified_files_count} 个章节文件中完成了角色名合并。")
        logger.info(f"成功自动重命名了 {renamed_wav_count} 个WAV音频文件。")

        # --- 5. 清理配置文件 (逻辑不变) ---
        if os.path.exists(profiles_path):
            with open(profiles_path, 'r', encoding='utf-8') as f: profiles = json.load(f)
            for name in source_names:
                if name in profiles: del profiles[name]
            with open(profiles_path, 'w', encoding='utf-8') as f: json.dump(profiles, f, ensure_ascii=False, indent=4)
            logger.info("已从角色简介中移除被合并的角色。")

        if character_timbres: # 使用已经加载的音色配置
            if target_name not in character_timbres:
                for name in source_names:
                    if name in character_timbres:
                        character_timbres[target_name] = character_timbres[name]
                        logger.info(f"目标角色 '{target_name}' 继承了源角色 '{name}' 的音色。")
                        break
            for name in source_names:
                if name in character_timbres: del character_timbres[name]
            with open(timbres_path, 'w', encoding='utf-8') as f: json.dump(character_timbres, f, ensure_ascii=False, indent=2)
            logger.info("已从音色配置中移除被合并的角色。")

        return {
            "status": "success", 
            "message": f"成功合并角色。{renamed_wav_count}个音频文件被自动重命名，无需重新生成。"
        }

    except Exception as e:
        logger.error(f"合并角色时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器处理合并时发生错误: {e}")
        
def clean_json_content(json_content: List[Dict]) -> List[Dict]:
    """
    Removes entries from the chapter JSON content if their 'content' field
    consists only of punctuation and whitespace.
    """
    # 定义一个包含中英文常用标点符号的集合
    # string.punctuation 包含 '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    # 我们再额外补充中文标点和一些特殊符号
    punctuation_to_remove = set(string.punctuation) | set("。，、；：？！—…“”《》‘’（）·")

    cleaned_list = []
    for item in json_content:
        content = item.get('content', '').strip()
        
        if not content:
            # 如果内容为空或只有空格，直接跳过
            continue

        # 移除所有定义的标点符号
        content_without_punctuation = ''.join(char for char in content if char not in punctuation_to_remove)
        
        # 再次去除可能剩余的空白（例如，如果原文是 "。 。"）
        if content_without_punctuation.strip():
            # 如果移除了标点和空格后，还有剩余内容，说明是有效语句，保留
            cleaned_list.append(item)
        else:
            # 否则，说明此行只包含标点，将其丢弃
            logger.info(f"正在清理无效语句: {item}")

    return cleaned_list
    
def validate_and_parse_json_array(text: str) -> Optional[list]:
    stripped_text = text.strip()
    if not (stripped_text.startswith('[') and stripped_text.endswith(']')):
        return None
    try:
        return json.loads(stripped_text)
    except json.JSONDecodeError:
        return None

def apply_replacement_rules(text: str, novel_name: str) -> str:
    """
    加载并应用小说专属替换词典中的规则。
    """
    replaced_text = text
    replace_dict_path = os.path.join(PROJECTS_DIR, novel_name, 'replace_dict.json')
    
    if os.path.exists(replace_dict_path):
        try:
            with open(replace_dict_path, 'r', encoding='utf-8') as f:
                replace_rules_data = json.load(f)
            rules = replace_rules_data.get("rules", [])
            
            # 对规则进行排序：从最长的 original_word 开始替换，避免短词影响长词
            rules.sort(key=lambda x: len(x.get("original_word", "")), reverse=True)
            
            applied_replacements = 0
            for rule in rules:
                original = rule.get("original_word")
                replacement = rule.get("replacement_word")
                if original and replacement:
                    new_text, num_replacements = re.subn(re.escape(original), replacement, replaced_text)
                    if num_replacements > 0:
                        replaced_text = new_text
                        applied_replacements += num_replacements
            
            if applied_replacements > 0:
                logger.debug(f"成功为小说「{novel_name}」应用了 {applied_replacements} 次替换规则 (TTS阶段)。")
            else:
                logger.debug(f"小说「{novel_name}」的替换词典中没有匹配到任何替换项 (TTS阶段)。")

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"加载或解析小说「{novel_name}」替换词典失败: {e}。将使用原始文本 (TTS阶段)。")
        except Exception as e:
            logger.error(f"应用替换规则时发生错误: {e}。将使用原始文本 (TTS阶段)。")
    else:
        logger.debug(f"replace_dict.json not found for novel '{novel_name}'. No replacements applied (TTS阶段).")

    return replaced_text
    
async def generate_chapter_json(chapter_content: str, model_name: str) -> List[Dict]:
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            llm_config = json.load(f)
    except FileNotFoundError:
        raise Exception("LLM配置文件丢失，无法执行操作。")

    model_settings = llm_config.get("models", {}).get(model_name)
    if not model_settings: raise ValueError(f"不支持的模型: {model_name}")

    api_key = model_settings.get("api_key")
    if not api_key: raise Exception(f"模型 '{model_name}' 的 API Key 未在配置文件中设置。")

    proxies = None
    proxy_config = llm_config.get("general", {}).get("proxy", {})
    if model_settings.get("use_proxy") and proxy_config.get("enabled"):
        p_addr = f"{proxy_config.get('protocol', 'socks5h')}://{proxy_config.get('address')}:{proxy_config.get('port')}"
        proxies = {"http": p_addr, "https": p_addr}
        logger.info(f"为模型 {model_name} 启用代理: {p_addr}")

    max_chars = model_settings.get("max_chars", 5000)
    actual_model_name = model_settings.get("model_name") 
    if not actual_model_name:
        raise Exception(f"模型 '{model_name}' 的 model_name 未在配置文件中设置。")

    if "aliyun" in model_name.lower(): # 用新的键名 'aliyun' 判断
        return await generate_with_qwen(chapter_content, actual_model_name, api_key, max_chars, proxies)
        
    elif "gemini" in model_name.lower():
        return await generate_with_gemini(chapter_content, actual_model_name, api_key, max_chars, proxies)
    else:
        raise ValueError(f"不支持的模型: {model_name}")
        
        
async def analyze_character(character_name: str, context_text: str, model_name_to_use: str) -> Optional[Dict]:
    """
    Analyzes a character based on text context using the specified model.
    """
    analysis_prompt = f"""请根据以下小说文本片段，深入分析角色 “{character_name}” 的人物特征。
你的任务是提取关键信息，并只返回一个严格符合以下格式的JSON对象，不要包含任何额外的解释或Markdown标记。
JSON对象必须包含以下三个键：
- "gender": (string) 角色的性别，推测为 "男", "女", 或 "未知"。
- "ageGroup": (string) 角色的年龄段，从 "孩童", "少年", "青年", "中年", "老年" 中选择一个最贴切的。
- "identity": (string) 角色的身份背景、职业、性格、出现场景、与其他角色的关系等和描述，以便后续帮助判断角色年龄，和其他角色是不是同一个人，200个字以内。
请确保你的回复中只包含这一个JSON对象。

文本片段:
---
{context_text[:2000]}
---
"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: config = json.load(f)
    except FileNotFoundError: return None

    # --- 核心修改：使用传入的 model_name_to_use ---
    model_settings = config.get("models", {}).get(model_name_to_use)
    if not model_settings:
        logger.error(f"分析角色时未找到模型 '{model_name_to_use}' 的配置。")
        return None

    api_key = model_settings.get("api_key")
    actual_model_name = model_settings.get("model_name")
    if not api_key or not actual_model_name:
        logger.error(f"模型 '{model_name_to_use}' 的 API Key 或 model_name 未配置。")
        return None

    proxies = None
    if model_settings.get("use_proxy") and config.get("general", {}).get("proxy", {}).get("enabled"):
        p_cfg = config["general"]["proxy"]
        p_addr = f"{p_cfg.get('protocol', 'socks5h')}://{p_cfg.get('address')}:{p_cfg.get('port')}"
        proxies = {"http": p_addr, "https": p_addr}

    # 根据模型平台选择不同的API URL和Payload
    api_url = ""
    headers = {}
    payload = {}

    if "gemini" in model_name_to_use.lower():
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{actual_model_name}:generateContent"
        headers = {'Content-Type': 'application/json', 'x-goog-api-key': api_key}
        payload = {"contents": [{"parts": [{"text": analysis_prompt}]}], "safetySettings": SAFETY_SETTINGS, "generationConfig": {"response_mime_type": "application/json"}}
    elif "aliyun" in model_name_to_use.lower():
        api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": actual_model_name,
            "input": {"messages": [{"role": "system", "content": "You are a helpful assistant that strictly follows user instructions to return JSON objects."}, {"role": "user", "content": analysis_prompt}]},
            "parameters": {"result_format": "message"}
        }
    else:
        logger.error(f"不支持的模型平台用于角色分析: {model_name_to_use}")
        return None

    try:
        response = requests.post(api_url, headers=headers, json=payload, proxies=proxies, timeout=120)
        response.raise_for_status()
        response_data = response.json()

        api_text = ""
        if "gemini" in model_name_to_use.lower():
            api_text = response_data['candidates'][0]['content']['parts'][0]['text']
        elif "aliyun" in model_name_to_use.lower():
            api_text = response_data["output"]["choices"][0]["message"]["content"]

        for match in re.finditer(r'\{.*?\}', api_text, re.DOTALL):
            try:
                profile = json.loads(match.group(0))
                if "gender" in profile and "identity" in profile:
                    logger.info(f"成功使用模型 '{actual_model_name}' 为 '{character_name}' 解析出简介。")
                    return profile
            except json.JSONDecodeError:
                continue

        logger.warning(f"未能从AI为 '{character_name}' 返回的文本中解析出有效的简介。")
        return None
    except Exception as e:
        logger.error(f"Request for character '{character_name}' profile failed: {e}")
        return None
        
async def complete_character_profile(character_name: str, existing_profile: Dict, context_text: str, model_name_to_use: str) -> Optional[Dict]:
    """
    Attempts to complete missing fields in an existing character profile using the specified model.
    """
    missing_fields = [key for key, value in existing_profile.items() if not value or value in ["未知", "Unknown"]]
    if not missing_fields:
        return None

    completion_prompt = f"""
已知角色 “{character_name}” 的部分信息如下：
{json.dumps(existing_profile, ensure_ascii=False, indent=2)}

其中，以下字段信息未知或不完整: {', '.join(missing_fields)}。
请根据以下小说文本片段，深入分析并尽力补全这些未知字段。
你的任务是返回一个完整的、只包含 "gender", "ageGroup", "identity" 三个键的JSON对象。即使某些字段仍然无法确定，也请返回一个完整的JSON结构，并将不确定的值设为 "未知"。
不要包含任何额外的解释或Markdown标记。

文本片段:
---
{context_text[:2500]} 
---
"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: config = json.load(f)
    except FileNotFoundError: return None
    
    model_settings = config.get("models", {}).get(model_name_to_use)
    if not model_settings:
        logger.error(f"补全角色简介时未找到模型 '{model_name_to_use}' 的配置。")
        return None

    api_key = model_settings.get("api_key")
    actual_model_name = model_settings.get("model_name")
    if not api_key or not actual_model_name:
        logger.error(f"模型 '{model_name_to_use}' 的 API Key 或 model_name 未配置。")
        return None

    proxies = None
    if model_settings.get("use_proxy") and config.get("general", {}).get("proxy", {}).get("enabled"):
        p_cfg = config["general"]["proxy"]
        p_addr = f"{p_cfg.get('protocol', 'socks5h')}://{p_cfg.get('address')}:{p_cfg.get('port')}"
        proxies = {"http": p_addr, "https": p_addr}
    
    api_url, headers, payload = "", {}, {}
    if "gemini" in model_name_to_use.lower():
        api_url = f"https://generativelaoguage.googleapis.com/v1beta/models/{actual_model_name}:generateContent"
        headers = {'Content-Type': 'application/json', 'x-goog-api-key': api_key}
        payload = {"contents": [{"parts": [{"text": completion_prompt}]}], "safetySettings": SAFETY_SETTINGS, "generationConfig": {"response_mime_type": "application/json"}}
    elif "aliyun" in model_name_to_use.lower():
        api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": actual_model_name,
            "input": {"messages": [{"role": "system", "content": "You are a helpful assistant that strictly follows user instructions to return JSON objects."}, {"role": "user", "content": completion_prompt}]},
            "parameters": {"result_format": "message"}
        }
    else:
        return None
    
    logger.info(f"    - Attempting to complete profile for '{character_name}', missing: {missing_fields}")
    try:
        response = requests.post(api_url, headers=headers, json=payload, proxies=proxies, timeout=120)
        response.raise_for_status()
        response_data = response.json()
        
        api_text = ""
        if "gemini" in model_name_to_use.lower():
            api_text = response_data['candidates'][0]['content']['parts'][0]['text']
        elif "aliyun" in model_name_to_use.lower():
            api_text = response_data["output"]["choices"][0]["message"]["content"]

        match = re.search(r'\{.*\}', api_text, re.DOTALL)
        if match:
            completed_data = json.loads(match.group(0))
            updated_profile = existing_profile.copy()
            for key, value in completed_data.items():
                if key in missing_fields and value and value not in ["未知", "Unknown"]:
                    updated_profile[key] = value
                    logger.info(f"      - Field '{key}' for '{character_name}' updated to '{value}'.")
            return updated_profile
        return None
    except Exception as e:
        logger.error(f"Request for completing '{character_name}' profile failed: {e}")
        return None
        
        
@app.post("/api/deep_analyze_character")
async def deep_analyze_character(req: DeepAnalyzeRequest):
    project_dir = os.path.join(PROJECTS_DIR, req.novel_name)
    json_dir = os.path.join(project_dir, 'chapters_json')
    profiles_path = os.path.join(project_dir, 'character_profiles.json')

    if not os.path.exists(profiles_path):
        raise HTTPException(status_code=404, detail="角色简介文件未找到。")
    
    with open(profiles_path, 'r', encoding='utf-8') as f:
        character_profiles = json.load(f)

    existing_profile = character_profiles.get(req.character_name)
    if not existing_profile:
        raise HTTPException(status_code=404, detail=f"未找到角色 '{req.character_name}' 的简介。")

    # --- 核心逻辑：聚合所有相关上下文 ---
    aggregated_context = ""
    if os.path.isdir(json_dir):
        for filename in os.listdir(json_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(json_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f_chap:
                    chapter_content = json.load(f_chap)
                
                # 提取该角色在此章节的所有对话
                dialogues = [item['content'] for item in chapter_content if item.get('speaker') == req.character_name]
                if dialogues:
                    aggregated_context += f"在章节《{filename.replace('.json','')}》中说到：\n" + "\n".join(dialogues) + "\n\n"
    
    if not aggregated_context:
        return {"status": "info", "message": "在已处理章节中未找到该角色的更多信息。"}

    # --- 调用我们预留的函数 ---
    updated_profile = await complete_character_profile(
        character_name=req.character_name,
        existing_profile=existing_profile,
        context_text=aggregated_context,
        model_name_to_use=req.model_name
    )

    if updated_profile:
        character_profiles[req.character_name] = updated_profile
        with open(profiles_path, 'w', encoding='utf-8') as f:
            json.dump(character_profiles, f, ensure_ascii=False, indent=4)
        return {"status": "success", "message": f"角色「{req.character_name}」的信息已成功补全！", "data": updated_profile}
    else:
        return {"status": "info", "message": "AI未能从现有信息中推断出更多内容。"}
        
@app.get("/api/get_novel_details")
async def get_novel_details(novel_name: str):
    """
    Gets a detailed list of all chapters from the source .txt for a novel,
    marks their processed status, and ALSO returns all character profiles.
    """
    project_dir = os.path.join(PROJECTS_DIR, novel_name)
    source_path = os.path.join(project_dir, 'source.txt')
    json_dir = os.path.join(project_dir, 'chapters_json')
    profiles_path = os.path.join(project_dir, 'character_profiles.json') # <-- 新增路径定义

    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail=f"Novel source file not found for '{novel_name}'")

    try:
        # --- 1. 获取章节信息 (逻辑不变) ---
        with open(source_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            content = f.read()
        
        all_chapters_from_txt = get_chapters_from_txt(content)
        
        processed_jsons = set()
        if os.path.isdir(json_dir):
            processed_jsons = {f for f in os.listdir(json_dir) if f.endswith('.json')}
        
        chapter_details = []
        for chap in all_chapters_from_txt:
            safe_title = "".join(c for c in chap['title'] if c.isalnum() or c in " _-").rstrip()
            is_processed = f"{safe_title}.json" in processed_jsons
            chapter_details.append({"title": chap['title'], "processed": is_processed})
        
        # --- 2. 新增：获取角色简介信息 ---
        character_profiles = {}
        if os.path.exists(profiles_path):
            try:
                with open(profiles_path, 'r', encoding='utf-8') as f:
                    character_profiles = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"无法读取或解析角色简介文件 '{profiles_path}': {e}")
                character_profiles = {} # 出错则返回空对象
        
        # --- 3. 将两部分信息合并返回 ---
        return {"chapters": chapter_details, "profiles": character_profiles}

    except Exception as e:
        logger.error(f"Error processing details for novel '{novel_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Error reading novel details: {e}")
        
# 差异点：为 /api/get_novel_content 接口添加禁止缓存的响应头
@app.get("/api/get_novel_content")
async def get_novel_content(filepath: str):
    """
    Serves the content of a specific processed chapter JSON file,
    with cache-control headers to prevent stale data.
    """
    try:
        path_parts = filepath.split('/', 1)
        if len(path_parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid filepath format.")
        
        novel_name, chapter_filename = path_parts
        full_path = os.path.join(PROJECTS_DIR, novel_name, 'chapters_json', chapter_filename)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse filepath.")

    project_root = os.path.abspath(PROJECTS_DIR)
    if not os.path.abspath(full_path).startswith(project_root):
        raise HTTPException(status_code=403, detail="禁止访问。")

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail=f"小说章节文件未找到: {filepath}")
    
    # Define headers that tell the browser not to cache this response.
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }
    # Return the FileResponse with the custom headers.
    return FileResponse(full_path, headers=headers)
    
# main.py -> HELPER FUNCTIONS
@app.delete("/api/delete_novel")
async def delete_novel(novel_name: str):
    """
    Deletes a novel project, including its project files and output audio.
    Includes comprehensive security checks.
    """
    if not novel_name:
        raise HTTPException(status_code=400, detail="小说名称不能为空。")

    # --- 核心安全检查：防止目录遍历攻击 ---
    # 确保 novel_name 是一个纯粹的目录名，不包含任何路径分隔符
    if novel_name != os.path.basename(novel_name) or ".." in novel_name:
        logger.warning(f"潜在的目录遍历攻击被阻止: {novel_name}")
        raise HTTPException(status_code=403, detail="非法的小说名称。")

    # --- 定位要删除的目录 ---
    project_dir = os.path.join(PROJECTS_DIR, novel_name)
    output_dir = os.path.join(OUTPUT_DIR, novel_name)

    # 检查项目目录是否存在，这是删除的必要条件
    if not os.path.isdir(project_dir):
        raise HTTPException(status_code=404, detail=f"小说项目 '{novel_name}' 未找到。")

    # --- 执行删除操作 ---
    errors = []
    # 1. 删除项目文件目录
    try:
        shutil.rmtree(project_dir)
        logger.info(f"成功删除项目目录: {project_dir}")
    except Exception as e:
        error_msg = f"删除项目目录 '{project_dir}' 失败: {e}"
        logger.error(error_msg)
        errors.append(error_msg)

    # 2. 删除输出文件目录 (如果存在)
    if os.path.isdir(output_dir):
        try:
            shutil.rmtree(output_dir)
            logger.info(f"成功删除输出目录: {output_dir}")
        except Exception as e:
            error_msg = f"删除输出目录 '{output_dir}' 失败: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    if errors:
        # 如果有任何一个删除操作失败，都返回一个错误
        raise HTTPException(status_code=500, detail="删除过程中发生错误: " + "; ".join(errors))

    return {"status": "success", "message": f"小说项目 '{novel_name}' 已被永久删除。"}
    
def smart_chunk_text(text: str, max_length: int) -> List[str]:
    """
    Splits a long text into smaller chunks without breaking sentences.
    """
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        # 确定当前块的最大结束位置
        end_pos = min(current_pos + max_length, len(text))
        
        # 如果已经到达文本末尾，直接添加剩余部分
        if end_pos == len(text):
            chunks.append(text[current_pos:])
            break
            
        # 从最大结束位置向前查找最佳分割点
        # 分割点的优先级：换行符 > 句号 > 感叹号 > 问号
        split_delimiters = ['\n', '。', '！', '？', '；', '，']
        best_split_pos = -1

        for delimiter in split_delimiters:
            # rfind 在指定范围内从右向左查找
            found_pos = text.rfind(delimiter, current_pos, end_pos)
            if found_pos != -1:
                best_split_pos = found_pos + 1 # 分割点在标点之后
                break
        
        # 如果找不到任何合适的标点，则强制在 max_length 处分割
        if best_split_pos == -1:
            best_split_pos = end_pos
            
        # 添加块到列表，并更新当前位置
        chunks.append(text[current_pos:best_split_pos])
        current_pos = best_split_pos
        
    logger.info(f"文本被智能分割为 {len(chunks)} 个块。")
    return chunks
    
async def normalize_character_names(new_names: List[str], existing_characters_with_profiles: Dict, context_text: str, model_name_to_use: str) -> Dict[str, str]:
    if not new_names or not existing_characters_with_profiles:
        return {}
    
    existing_names_formatted = json.dumps(existing_characters_with_profiles, ensure_ascii=False, indent=2)

    normalization_prompt = f"""你是一个专业的小说编辑，擅长根据人物简介和上下文，识别角色的不同称谓。
你的任务是判断“新出现的名字”是否是“已存在角色”的别名。

---
【已知信息】

1. 已存在角色的简介:
{existing_names_formatted}

2. 新名字出现的章节上下文:
{context_text[:2500]}
---

【待判断】

新出现的名字列表:
{json.dumps(new_names, ensure_ascii=False)}

---
【任务要求】

1. **综合分析**: 仔细阅读【已知信息】，判断“新出现的名字”是否指代“已存在角色”。
   - 依据包括但不限于：姓名关联性（如“张真人” -> “张三丰”）、上下文中的行为、对话、他人对他们的称呼等。
   - 特别注意人物简介中的“性别”、“年龄段”、“身份”等关键特征是否匹配。
2. **严格的JSON输出**: 你的回答必须是一个严格的JSON对象，代表一个从“新名字”到“已存在角色名”的映射。
3. **只映射确定的别名**: 如果一个新名字无法【非常有信心地】确定是任何已存在角色的别名，则【绝对不要】在JSON中包含它。宁可漏掉，不可错判。
4. **空结果**: 如果没有任何新名字是别名，则必须返回一个空的JSON对象 `{{}}`。
5. **无额外文本**: 除了JSON对象，不要输出任何其他解释、说明或Markdown标记。

例如，如果判断“王姑娘”的言行和简介都符合“王语嫣”，则应返回：
`{{"王姑娘": "王语嫣"}}`
"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: config = json.load(f)
    except FileNotFoundError: return {}
    
    model_settings = config.get("models", {}).get(model_name_to_use)
    if not model_settings: return {}
    
    api_key = model_settings.get("api_key")
    actual_model_name = model_settings.get("model_name")
    if not api_key or not actual_model_name: return {}
    
    proxies = None
    if model_settings.get("use_proxy") and config.get("general", {}).get("proxy", {}).get("enabled"):
        p_cfg = config["general"]["proxy"]
        p_addr = f"{p_cfg.get('protocol', 'socks5h')}://{p_cfg.get('address')}:{p_cfg.get('port')}"
        proxies = {"http": p_addr, "https": p_addr}

    api_url, headers, payload = "", {}, {}
    if "gemini" in model_name_to_use.lower():
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{actual_model_name}:generateContent"
        headers = {'Content-Type': 'application/json', 'x-goog-api-key': api_key}
        payload = {"contents": [{"parts": [{"text": normalization_prompt}]}], "safetySettings": SAFETY_SETTINGS, "generationConfig": {"response_mime_type": "application/json"}}
    elif "aliyun" in model_name_to_use.lower():
        api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": actual_model_name,
            "input": {"messages": [{"role": "system", "content": "You are a helpful assistant that strictly follows user instructions to return JSON objects."}, {"role": "user", "content": normalization_prompt}]},
            "parameters": {"result_format": "message"}
        }
    else:
        return {}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, proxies=proxies, timeout=180)
        response.raise_for_status()
        response_data = response.json()

        api_text = ""
        if "gemini" in model_name_to_use.lower():
            api_text = response_data['candidates'][0]['content']['parts'][0]['text']
        elif "aliyun" in model_name_to_use.lower():
            api_text = response_data["output"]["choices"][0]["message"]["content"]
        
        match = re.search(r'\{.*\}', api_text, re.DOTALL)
        if match:
            mapping = json.loads(match.group(0))
            existing_names = list(existing_characters_with_profiles.keys())
            cleaned_mapping = {k: v for k, v in mapping.items() if k in new_names and v in existing_names}
            logger.info(f"角色名归一化映射结果: {cleaned_mapping}")
            return cleaned_mapping
        return {}
    except Exception as e:
        logger.error(f"角色名归一化失败: {e}")
        return {}
        
@app.post("/api/process_single_chapter")
async def process_single_chapter(req: ProcessSingleChapterRequest):
    """
    Processes a single chapter.
    If preview_only is true, returns the raw text content without processing.
    If force_regenerate is true, deletes existing audio files first.
    """
    project_dir = os.path.join(PROJECTS_DIR, req.novel_name)
    source_path = os.path.join(project_dir, 'source.txt')
    json_dir = os.path.join(project_dir, 'chapters_json') # <-- 添加这一行
    profiles_path = os.path.join(project_dir, 'character_profiles.json') # <-- 添加这一行
    
    # --- 1. 加载资源 (提前) ---
    try:
        with open(source_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            content = f.read()
        all_chapters_map = {c['title']: c['content'] for c in get_chapters_from_txt(content)}
        chapter_content = all_chapters_map.get(req.chapter_title)

        if not chapter_content:
            raise HTTPException(status_code=404, detail=f"Chapter '{req.chapter_title}' not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取项目文件失败: {e}")

    # --- 2. 新增：预览模式逻辑 ---
    if req.preview_only:
        logger.info(f"为章节 '{req.chapter_title}' 提供原文预览。")
        return {"status": "preview", "content": chapter_content}
    # ---

    # 3. 检查处理模式所必需的 model_name
    if not req.model_name:
        raise HTTPException(status_code=400, detail="处理章节需要提供 model_name。")

    # --- 新增：文件清理逻辑 ---
    if req.force_regenerate:
        logger.warning(f"强制重新生成模式已激活，将为章节 '{req.chapter_title}' 清理旧文件。")
        safe_title = "".join(c for c in req.chapter_title if c.isalnum() or c in " _-").rstrip()
        
        # 1. 删除单句 WAV 文件目录
        chapter_wav_dir = os.path.join(OUTPUT_DIR, req.novel_name, 'wavs', safe_title)
        if os.path.isdir(chapter_wav_dir):
            try:
                shutil.rmtree(chapter_wav_dir)
                logger.info(f"  - 已删除目录: {chapter_wav_dir}")
            except Exception as e:
                logger.error(f"  - 删除目录 {chapter_wav_dir} 失败: {e}")

        # 2. 删除最终拼接的音频文件
        final_audio_dir = os.path.join(OUTPUT_DIR, req.novel_name)
        if os.path.isdir(final_audio_dir):
            for f in os.listdir(final_audio_dir):
                if f.startswith(safe_title):
                    file_to_delete = os.path.join(final_audio_dir, f)
                    try:
                        os.remove(file_to_delete)
                        logger.info(f"  - 已删除文件: {file_to_delete}")
                    except Exception as e:
                        logger.error(f"  - 删除文件 {file_to_delete} 失败: {e}")
    # --- 文件清理逻辑结束 ---

    try:
        with open(source_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            content = f.read()
        all_chapters_map = {c['title']: c['content'] for c in get_chapters_from_txt(content)}
        
        # 注意：这里读取 profiles 文件时，即使文件不存在也应该继续，而不是抛出异常
        try:
            with open(profiles_path, 'r', encoding='utf-8') as f:
                character_profiles = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            character_profiles = {}

    except Exception as e:
         raise HTTPException(status_code=500, detail=f"读取项目文件失败: {e}")

    existing_character_names = list(character_profiles.keys())
    title = req.chapter_title
    chapter_content = all_chapters_map.get(title)

    if not chapter_content:
        raise HTTPException(status_code=404, detail=f"Chapter '{title}' not found in source text.")

    logger.info(f"Processing single chapter: {title}")
    
    try:
        raw_json_content = await generate_chapter_json(chapter_content, req.model_name)
        full_json_content = clean_json_content(raw_json_content)
        speakers_in_chapter = sorted(list({item['speaker'] for item in full_json_content if 'speaker' in item and item['speaker'] != "旁白"}))
        new_potential_names = [name for name in speakers_in_chapter if name not in existing_character_names]

        #logger.info(f"new_potential_names: {new_potential_names}, existing_character_names: {existing_character_names}")
        #logger.info(f"character_profiles: {character_profiles}, new_potential_names: {new_potential_names}")
        name_mapping = {}
        if new_potential_names and character_profiles: # 仅当已有角色时才进行归一化
            name_mapping = await normalize_character_names(new_potential_names, character_profiles, chapter_content, req.model_name)

        if name_mapping:
            logger.info(f"应用名称映射: {name_mapping}")
            for item in full_json_content:
                if item.get('speaker') in name_mapping:
                    original_name = item['speaker']
                    new_name = name_mapping[original_name]
                    item['speaker'] = new_name
                    logger.info(f"  - 将 '{original_name}' 替换为 '{new_name}'")

        os.makedirs(json_dir, exist_ok=True) 
        safe_title = "".join(c for c in title if c.isalnum() or c in " _-").rstrip()
        output_path = os.path.join(json_dir, f"{safe_title}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_json_content, f, ensure_ascii=False, indent=2)
        logger.info(f"Chapter '{title}' successfully saved to: {output_path}")

        final_speakers_in_chapter = sorted(list({item['speaker'] for item in full_json_content if 'speaker' in item and item['speaker'] != "旁白"}))
        truly_new_characters = [name for name in final_speakers_in_chapter if name not in existing_character_names]
        
        newly_analyzed_count = 0
        if truly_new_characters:
            logger.info(f"发现真正的新角色: {', '.join(truly_new_characters)}")
            
            profiles_were_updated = False # 确保这个变量被定义
            # --- 核心修改：恢复为每个新角色独立、简单地处理 ---
            for char_name in truly_new_characters:
                context_for_analysis = chapter_content
                first_occurrence = chapter_content.find(char_name)
                
                if first_occurrence != -1:
                    context_window = 1200
                    start = max(0, first_occurrence - context_window)
                    end = min(len(chapter_content), first_occurrence + len(char_name) + context_window)
                    context_for_analysis = chapter_content[start:end]
                
                logger.info(f"为新角色 '{char_name}' 请求简介分析 (使用模型: {req.model_name})...")
                # 修改后的调用，传入 req.model_name
                profile = await analyze_character(char_name, context_for_analysis, req.model_name)
                
                if profile:
                    # 直接、简单地赋值
                    character_profiles[char_name] = profile
                    profiles_were_updated = True

            if profiles_were_updated:
                with open(profiles_path, 'w', encoding='utf-8') as f:
                    json.dump(character_profiles, f, ensure_ascii=False, indent=4)
                
        return {"status": "success", "message": f"章节 '{title}' 处理成功。"}

    except Exception as e:
        logger.error(f"Failed to process single chapter '{title}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process chapter '{title}': {e}")
        
# 差异点：替换 get_chapters_from_txt 函数的实现
# 差异点：使用全新的、基于启发式规则的 get_chapters_from_txt 函数

def get_chapters_from_txt(text_content: str) -> List[Dict]:
    """
    Intelligently splits a text into chapters using a weighted, heuristic-based
    engine to identify the dominant chapter pattern and filter out noise.
    """
    
    # === Stage 1: Candidate Generation & Feature Engineering ===

    # A comprehensive but tolerant regex to find ALL potential chapter-like lines.
    # It captures the entire line for analysis.
    candidate_regex = re.compile(
        r"^\s*("
        r"(?:第\s*[零〇一二三四五六七八九十百千万\d]+\s*[章回卷节])"  # e.g., 第一章, 第100回
        r"|"
        r"(?:[正外]篇|[上下]部)" # e.g., 正篇, 上部
        r"|"
        r"(?:\d{1,5})" # e.g., 101
        r"|"
        r"(?:[一二三四五六七八九十百千万零〇]+)" # e.g., 一百零一
        r"|"
        r"(?:[\(（\[【]\s*\d+\s*[\)）\]】])" # e.g., (101)
        r").*?$", re.MULTILINE
    )
    
    candidates = []
    # Use splitlines() to accurately determine if a line is standalone
    lines = text_content.splitlines()
    line_map = {line.strip(): i for i, line in enumerate(lines)}
    
    last_numeric_val = 0
    for match in candidate_regex.finditer(text_content):
        line_content = match.group(0).strip()
        
        # Feature Extraction
        features = {
            'text': line_content,
            'pos': match.start(),
            'pattern_type': None,
            'is_standalone': False,
            'length': len(line_content),
            'contains_chapter_word': any(kw in line_content for kw in ['章', '回', '卷', '节', '篇', '部']),
            'is_sequential': False,
            'numeric_val': 0
        }
        
        # Determine pattern type
        if re.search(r'^第\s*[\d一二三四五六七八九十百千万零〇]+\s*[章回卷节]', line_content):
            features['pattern_type'] = 'formal_chapter' # Highest weight
        elif re.search(r'^\d+', line_content):
            features['pattern_type'] = 'numeric_list'
        else:
            features['pattern_type'] = 'other'

        # Check if it's a standalone line (heuristic: next line is empty or doesn't start immediately)
        line_index = line_map.get(line_content)
        if line_index is not None and (line_index + 1 >= len(lines) or not lines[line_index + 1].strip()):
            features['is_standalone'] = True

        # Check for sequential numbering (heuristic)
        numeric_part = re.search(r'\d+', line_content)
        if numeric_part:
            current_numeric_val = int(numeric_part.group(0))
            if current_numeric_val == last_numeric_val + 1:
                features['is_sequential'] = True
            last_numeric_val = current_numeric_val
            features['numeric_val'] = current_numeric_val

        candidates.append(features)

    if not candidates:
        if text_content.strip():
            return [] # 直接返回空列表
        return []

    # === Stage 2: Weighted Scoring & Decision ===
    
    pattern_scores = {}
    for cand in candidates:
        score = 0
        # Weights - these can be tuned
        if cand['is_standalone']: score += 50
        if cand['length'] > 50: score -= 100 # Heavy penalty for long lines
        if cand['contains_chapter_word']: score += 20
        if cand['is_sequential']: score += 10
        
        # Base score for pattern type
        if cand['pattern_type'] == 'formal_chapter': score += 30
        
        pattern_scores.setdefault(cand['pattern_type'], []).append(score)

    # Calculate average score for each pattern type
    avg_scores = {
        pattern: sum(scores) / len(scores)
        for pattern, scores in pattern_scores.items()
        if scores
    }
    
    if not avg_scores: # If no patterns scored positively
        dominant_pattern = 'other' # Fallback
    else:
        # The pattern with the highest average score wins
        dominant_pattern = max(avg_scores, key=avg_scores.get)
    
    #logger.info(f"Dominant chapter pattern identified: {dominant_pattern} with scores: {avg_scores}")
    
    # === Stage 3: Precise Extraction ===
    
    # Filter candidates to only include those matching the dominant pattern
    final_titles = [cand for cand in candidates if cand['pattern_type'] == dominant_pattern and cand['length'] <= 50]

    if not final_titles:
        # Fallback if the dominant pattern was wrong or filtered out everything
        logger.warning("Dominant pattern resulted in no chapters. Falling back to simple extraction.")
        # As a simple fallback, let's use the original regex and split
        return [] # 直接返回空列表


    chapters = []
    # Handle content before the first real chapter
    first_chapter_pos = final_titles[0]['pos']
    if first_chapter_pos > 0:
        intro_content = text_content[:first_chapter_pos].strip()
        if intro_content:
            chapters.append({"title": "前言", "content": intro_content})

    for i, title_info in enumerate(final_titles):
        start_pos = title_info['pos']
        # End position is the start of the next chapter, or the end of the text
        end_pos = final_titles[i + 1]['pos'] if i + 1 < len(final_titles) else len(text_content)
        
        full_chapter_text = text_content[start_pos:end_pos].strip()
        
        # The full title is the first line of this chunk
        parts = full_chapter_text.split('\n', 1)
        full_title = parts[0].strip()
        content = parts[1].strip() if len(parts) > 1 else ""

        chapters.append({"title": full_title, "content": content})
        
    return chapters

@app.post("/api/apply_effect")
async def apply_audio_effect(req: EffectRequest):
    file_path = os.path.join(OUTPUT_DIR, req.novel_name, 'wavs', req.chapter_name, req.file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="音频文件未找到。")
    try:
        # 强制指定 frame_rate，避免 pydub 从文件中读取错误的值
        audio = AudioSegment.from_wav(file_path)
        processed_audio = None
        
        if req.effect_type == 'phone':
            # *** 核心修改：增强手机通话效果 ***
            
            # 1. 滤波，限制频率范围 (300Hz - 3400Hz)
            processed_audio = high_pass_filter(audio, 300)
            processed_audio = low_pass_filter(processed_audio, 3400)
            
            # 2. （新增）轻微增加音量，模拟压缩效果
            processed_audio = processed_audio + 3 
            
            # 3. （新增）降低采样率和比特深度，产生“lo-fi”感
            #    将采样率降至 8000 Hz，这是电话语音的标准
            processed_audio = processed_audio.set_frame_rate(8000)
            #    再升回原始采样率，pydub会自动进行重采样，这个过程会带来独特的数码感
            processed_audio = processed_audio.set_frame_rate(audio.frame_rate)
            
        elif req.effect_type == 'megaphone':
            # 增强喇叭效果
            processed_audio = high_pass_filter(audio, 500)
            processed_audio = low_pass_filter(processed_audio, 5000)
            # 增加一点失真感，通过轻微的过载
            processed_audio = processed_audio.apply_gain_stereo(+6).compress_dynamic_range(threshold=-10.0)
            
        elif req.effect_type == 'reverb':
            # 增强混响效果
            # 创建一个更弱、更延迟的回声
            reverb_audio_1 = audio - 18
            reverb_audio_2 = audio - 24
            # 混合主音轨和两个延迟的回声
            processed_audio = audio.overlay(reverb_audio_1, position=150)
            processed_audio = processed_audio.overlay(reverb_audio_2, position=300)
            
        else:
            raise HTTPException(status_code=400, detail="未知的特效类型。")

        if processed_audio:
            # 导出前进行标准化，防止削波
            processed_audio = normalize(processed_audio)
            processed_audio.export(file_path, format="wav")
            # 返回一个更具体的成功消息
            effect_name_map = {"phone": "手机通话", "megaphone": "喇叭喊话", "reverb": "室内回声"}
            return {"status": "success", "message": f"'{effect_name_map.get(req.effect_type, req.effect_type)}' 特效已应用。"}
        else:
            raise Exception("处理音频失败。")
            
    except Exception as e:
        logger.error(f"处理特效失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器处理特效失败: {e}")

def trim_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    """
    Trims silence from the beginning of a Pydub AudioSegment.
    """
    trim_ms = detect_leading_silence(
        sound,
        silence_threshold=silence_threshold,
        chunk_size=chunk_size
    )
    return sound[trim_ms:]

@app.post("/api/generate_choral_effect")
async def generate_choral_effect(req: ChoralRequest):
    if len(req.selected_timbres) < 2:
        raise HTTPException(status_code=400, detail="请至少选择两个音色。")

    # 1. 加载配置并确定 TTS 服务 endpoint
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: config = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载配置文件失败: {e}")

    tts_models_config = config.get("tts_models", {})
    default_tts_model = config.get("general", {}).get("default_tts_model")
    model_id_to_use = req.tts_model if req.tts_model in tts_models_config else default_tts_model
    
    if not model_id_to_use or model_id_to_use not in tts_models_config:
        raise HTTPException(status_code=400, detail="未找到有效的目标 TTS 模型配置。")
    
    model_endpoint = tts_models_config[model_id_to_use].get("endpoint")
    if not model_endpoint:
        raise HTTPException(status_code=500, detail=f"模型 '{model_id_to_use}' 未配置 endpoint。")

    # 2. 为每个音色调用 TTS 微服务生成单人语音
    request_temp_dir = os.path.join(TEMP_DIR, f"choral_{uuid.uuid4()}")
    os.makedirs(request_temp_dir, exist_ok=True)
    generated_wav_paths = []
    
    try:
        for timbre_name in req.selected_timbres:
            logger.info(f"为合声效果生成音色 '{timbre_name}' (使用模型: {model_id_to_use})")
            
            # a. 加载参考音频和文本
            timbre_dir = os.path.join(WAV_DIR, timbre_name)
            prompt_wav_path = os.path.join(timbre_dir, "1.wav")
            prompt_txt_path = os.path.join(timbre_dir, "1.txt")
            if not (os.path.exists(prompt_wav_path) and os.path.exists(prompt_txt_path)):
                logger.warning(f"音色 '{timbre_name}' 文件不完整，已跳过。")
                continue
            
            with open(prompt_wav_path, "rb") as f_wav:
                prompt_audio_b64 = base64.b64encode(f_wav.read()).decode('utf-8')
            with open(prompt_txt_path, 'r', encoding='utf-8') as f_txt:
                prompt_text = f_txt.read()

            # b. 构建 payload 并调用微服务
            payload = {
                "tts_text": req.tts_text,
                "prompt_audio": prompt_audio_b64,
                "prompt_text": prompt_text
            }
            response = requests.post(model_endpoint, json=payload, timeout=300)
            response.raise_for_status()
            tts_response_data = response.json()

            if tts_response_data.get("status") == "success" and tts_response_data.get("audio"):
                audio_data = base64.b64decode(tts_response_data["audio"])
                temp_wav_path = os.path.join(request_temp_dir, f"{timbre_name}.wav")
                with open(temp_wav_path, "wb") as f_out:
                    f_out.write(audio_data)
                generated_wav_paths.append(temp_wav_path)
            else:
                logger.warning(f"音色 '{timbre_name}' 生成失败: {tts_response_data.get('message')}")

        if not generated_wav_paths:
            raise Exception("所有选定音色的语音均生成失败。")

        # 3. 混合音频 (逻辑不变)
        logger.info(f"正在使用高级混合技术处理 {len(generated_wav_paths)} 个音轨...")
        segments = [AudioSegment.from_wav(p) for p in generated_wav_paths]
        max_duration = max(s.duration_seconds for s in segments)
        canvas = AudioSegment.silent(duration=int(max_duration * 1000) + 100, frame_rate=segments[0].frame_rate)
        canvas = canvas.set_channels(2)
        for segment in segments:
            if segment.channels == 1: segment = segment.set_channels(2)
            random_gain = -6 - random.uniform(0, 4)
            processed_segment = segment.apply_gain(random_gain).pan(random.uniform(-0.8, 0.8))
            canvas = canvas.overlay(processed_segment, position=random.randint(5, 30))

        # 4. 保存最终文件
        final_audio = normalize(canvas)
        safe_speaker = "".join(c for c in req.original_speaker if c.isalnum() or c in " _-").rstrip()
        safe_timbre = "".join(c for c in req.original_timbre if c.isalnum() or c in " _-").rstrip()
        output_wav_name = f"{req.row_index:04d}-{safe_speaker}-{safe_timbre}.wav"
        wav_output_dir = os.path.join(OUTPUT_DIR, req.novel_name, 'wavs', req.chapter_name)
        output_full_path = os.path.join(wav_output_dir, output_wav_name)
        final_audio.export(output_full_path, format="wav")
        
        return {"status": "success", "message": "多人同声效果生成成功！", "file_name": output_wav_name}

    finally:
        if os.path.exists(request_temp_dir):
            shutil.rmtree(request_temp_dir)

            
# =================================================================
#               CORE API ENDPOINTS
# =================================================================
@app.post("/api/upload_txt_novel")
async def upload_txt_novel(file: UploadFile = File(...)):
    novel_name = os.path.splitext(file.filename)[0]
    project_dir = os.path.join(PROJECTS_DIR, novel_name)
    os.makedirs(project_dir, exist_ok=True)
    source_path = os.path.join(project_dir, 'source.txt')
    marker_path = os.path.join(project_dir, '.is_txt_project')
    
    try:
        content_bytes = await file.read()
        
        # --- 核心修复：优化解码逻辑，不依赖 chardet ---
        content_text = ""
        # 定义一个按成功率排序的编码尝试列表
        encodings_to_try = [
            'utf-8-sig',  # 优先处理带 BOM 的 UTF-8 (Windows 记事本常见)
            'utf-8',      # 标准 UTF-8
            'gb18030',    # 最宽容的中文编码，完全兼容 GBK 和 GB2312
            'big5'        # 备选：繁体中文编码
        ]
        
        for encoding in encodings_to_try:
            try:
                content_text = content_bytes.decode(encoding)
                logger.info(f"成功使用编码 '{encoding}' 解码上传的文件 '{file.filename}'。")
                break  # 解码成功，立即跳出循环
            except (UnicodeDecodeError, TypeError):
                logger.warning(f"尝试使用编码 '{encoding}' 解码失败，继续尝试下一个。")
                continue # 继续尝试列表中的下一个编码
        
        # 如果所有尝试都失败了，执行最终的回退方案
        if not content_text:
            content_text = content_bytes.decode('utf-8', errors='replace')
            logger.error(f"所有编码尝试均失败，已为文件 '{file.filename}' 强制替换未知字符。")
        # --- 修复结束 ---

        # 将正确解码的 Unicode 字符串，统一以标准的 UTF-8 格式写入文件
        with open(source_path, "w", encoding="utf-8") as buffer:
            buffer.write(content_text)
        
        with open(marker_path, 'w') as f:
            f.write('')
        
        chapters = get_chapters_from_txt(content_text)
        
        chapters_cache_path = os.path.join(project_dir, 'chapters_cache.json')
        chapters_to_cache = [{"title": chap["title"]} for chap in chapters]
        with open(chapters_cache_path, 'w', encoding='utf-8') as f:
            json.dump(chapters_to_cache, f, ensure_ascii=False, indent=2)
        logger.info(f"章节列表已缓存到: {chapters_cache_path}")

        chapters_for_frontend = [{"id": i, "title": chap["title"]} for i, chap in enumerate(chapters)]

        return {
            "status": "success", 
            "message": f"小说 '{novel_name}' 已成功上传并统一转换为UTF-8。",
            "chapters": chapters_for_frontend
        }
    except Exception as e:
        # 清理可能已创建的文件
        if os.path.exists(project_dir):
            # 为了安全，这里可以选择性删除，或者在开发阶段保留以便调试
            # shutil.rmtree(project_dir) 
            pass
        logger.error(f"处理上传的TXT文件失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"保存或处理源文件失败: {e}")

@app.get("/api/list_novels")
async def list_novels():
    if not os.path.isdir(PROJECTS_DIR): return {"novels_details": {}}
    novels_details = {}
    novel_names = [d for d in os.listdir(PROJECTS_DIR) if os.path.isdir(os.path.join(PROJECTS_DIR, d))]
    
    for name in novel_names:
        project_dir = os.path.join(PROJECTS_DIR, name)
        source_path = os.path.join(project_dir, 'source.txt')
        json_dir = os.path.join(project_dir, 'chapters_json')
        output_novel_dir = os.path.join(OUTPUT_DIR, name)

        # --- 核心修改 3: 检查标记文件 ---
        is_txt_project = os.path.exists(os.path.join(project_dir, '.is_txt_project'))

        if not os.path.exists(source_path): continue
        
        try:
            with open(source_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                content = f.read()
            
            all_chapters_from_txt = get_chapters_from_txt(content)
            processed_jsons = set()
            if os.path.isdir(json_dir):
                processed_jsons = {f for f in os.listdir(json_dir) if f.endswith('.json')}
            
            spliced_audios = set()
            if os.path.isdir(output_novel_dir):
                spliced_audios = {f for f in os.listdir(output_novel_dir) if f.endswith(('.mp3', '.wav', '.m4a', '.ogg'))}

            chapter_details = []
            for i, chap in enumerate(all_chapters_from_txt):
                safe_title = "".join(c for c in chap['title'] if c.isalnum() or c in " _-").rstrip()
                is_processed = f"{safe_title}.json" in processed_jsons
                is_spliced = any(f.startswith(safe_title) for f in spliced_audios)
                
                # --- 核心修改 4: 统一返回结构 ---
                chapter_info = {
                    "id": i,  # 为TXT项目提供唯一ID
                    "title": chap['title'], 
                    "processed": is_processed, 
                    "spliced": is_spliced
                }
                chapter_details.append(chapter_info)
            
            # --- 核心修改 5: 将 isTxtProject 标记添加到响应中 ---
            novels_details[name] = {
                "chapters": chapter_details,
                "isTxtProject": is_txt_project
            }
        except Exception as e:
            logger.error(f"Error processing details for novel '{name}': {e}")
            continue
            
    return {"novels_details": novels_details}


@app.get("/api/get_character_profile")
async def get_character_profile(novel_name: str, character_name: str):
    decoded_char_name = urllib.parse.unquote(character_name)
    profiles_path = os.path.join(PROJECTS_DIR, novel_name, 'character_profiles.json')
    if not os.path.exists(profiles_path): raise HTTPException(status_code=404, detail="该小说的角色简介文件未找到。")
    try:
        with open(profiles_path, 'r', encoding='utf-8') as f: profiles = json.load(f)
        profile = profiles.get(decoded_char_name)
        if not profile: raise HTTPException(status_code=404, detail=f"未找到角色 '{decoded_char_name}' 的简介。")
        return profile
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get_novel_content")
async def get_novel_content(filepath: str):
    # filepath is expected to be novel_name/chapter_name.json
    file_path = os.path.join(PROJECTS_DIR, filepath)
    if not os.path.abspath(file_path).startswith(os.path.abspath(PROJECTS_DIR)):
        raise HTTPException(status_code=403, detail="禁止访问。")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="小说文件未找到。")
    return FileResponse(file_path)

@app.post("/api/get_characters_in_chapters")
async def get_characters_in_chapters(req: CharactersInChaptersRequest):
    if not req.chapter_files: return {"characters": []}
    all_speakers = set()
    json_dir = os.path.join(PROJECTS_DIR, req.novel_name, 'chapters_json')
    for chapter_file in req.chapter_files:
        file_path = os.path.join(json_dir, chapter_file)
        if not os.path.abspath(file_path).startswith(os.path.abspath(json_dir)): continue
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        if 'speaker' in item: all_speakers.add(item['speaker'])
        except Exception as e:
            logger.error(f"Error reading or parsing {file_path}: {e}")
            continue
    return {"characters": sorted(list(all_speakers))}

@app.get("/api/get_config")
async def get_config(novel_name: str):
    config_path = os.path.join(PROJECTS_DIR, novel_name, 'character_timbres.json')
    if not os.path.exists(config_path):
        return JSONResponse(content={})
    return FileResponse(config_path)

@app.post("/api/update_config")
async def update_config(req: UpdateConfigRequest):
    project_dir = os.path.join(PROJECTS_DIR, req.novel_name)
    if not os.path.isdir(project_dir):
        raise HTTPException(status_code=404, detail="小说项目未找到。")
    config_path = os.path.join(project_dir, 'character_timbres.json')
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(req.config_data, f, ensure_ascii=False, indent=2)
        return {"status": "success", "message": f"小说 '{req.novel_name}' 的音色配置已保存。"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器端写入文件失败: {e}")

@app.post("/api/tts_v2")
async def text_to_speech_v2(req: TTSRequestV2):
    try:
        # 1. 加载配置 (逻辑不变)
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 2. 确定要使用的 TTS 模型和其 endpoint (逻辑不变)
        tts_models_config = config.get("tts_models", {})
        default_tts_model = config.get("general", {}).get("default_tts_model")
        
        model_id_to_use = req.tts_model if req.tts_model in tts_models_config else default_tts_model
        
        if not model_id_to_use or model_id_to_use not in tts_models_config:
            raise HTTPException(status_code=400, detail="未找到有效的目标 TTS 模型配置。")
            
        model_endpoint = tts_models_config[model_id_to_use].get("endpoint")
        if not model_endpoint:
            raise HTTPException(status_code=500, detail=f"模型 '{model_id_to_use}' 未配置 endpoint。")

        processed_tts_text = apply_replacement_rules(req.tts_text, req.novel_name)
        
        # --- 3. 核心修复：构建包含所有模式信息的完整 payload ---
        payload = {
            "tts_text": processed_tts_text, # <-- 使用替换后的文本
            "prompt_audio": req.prompt_audio,
            "prompt_text": req.prompt_text,
            "inference_mode": req.inference_mode,    # <-- 新增：传递推理模式
            "instruct_text": req.instruct_text       # <-- 新增：传递指令文本
        }
        
        print("prompt is ", req.prompt_text)
            
        best_audio_data_for_saving = None
        min_tail_dbfs_found_in_retries = float('inf') # 初始化为正无穷大

        # NEW: 后端内部的重试循环
        for attempt in range(TTS_GENERATION_MAX_RETRIES):
            logger.info(f"正在向 TTS 服务 '{model_id_to_use}' ({model_endpoint}) 发送请求 (行: {req.row_index}, 尝试: {attempt + 1}/{TTS_GENERATION_MAX_RETRIES})，模式: '{req.inference_mode}'...")
            
            current_attempt_audio_data = None # 存储当前尝试获取的音频数据
            current_attempt_tail_dbfs = float('inf') # 存储当前尝试的结尾能量

            try:
                response = requests.post(model_endpoint, json=payload, timeout=300)
                response.raise_for_status() # 如果状态码不是 2xx，则抛出异常
                
                tts_response_data = response.json()
                if tts_response_data.get("status") != "success":
                    logger.warning(f"TTS 服务返回错误 (尝试: {attempt + 1}): {tts_response_data.get('message', '未知错误')}")
                    await asyncio.sleep(2) # 短暂等待后重试
                    continue # 继续下一次重试

                audio_data_b64 = tts_response_data.get("audio")
                if not audio_data_b64:
                    logger.warning(f"TTS 服务响应中缺少音频数据 (尝试: {attempt + 1})。")
                    await asyncio.sleep(2) # 短暂等待后重试
                    continue # 继续下一次重试
                    
                current_attempt_audio_data = base64.b64decode(audio_data_b64)

                # NEW: 音频结尾能量判断逻辑
                try:
                    audio_segment = AudioSegment.from_file(io.BytesIO(current_attempt_audio_data), format="wav")
                    audio_duration_ms = len(audio_segment) # 获取音频时长（毫秒）

                    # 安全检查：对于极短音频，不进行结尾能量分析，直接认为成功
                    if audio_duration_ms < TTS_TAIL_ANALYSIS_DURATION_MS:
                        logger.info(f"  TTS音频极短 (总时长 {audio_duration_ms}ms)，跳过结尾能量分析。认为正常。")
                        
                        # 视为最佳结果，直接跳出重试循环
                        best_audio_data_for_saving = current_attempt_audio_data
                        break 
                    
                    # 提取音频结尾片段并计算能量
                    tail_segment = audio_segment[-TTS_TAIL_ANALYSIS_DURATION_MS:]
                    current_attempt_tail_dbfs = tail_segment.dBFS
                    
                    logger.info(f"  TTS音频时长检查: 文本长度 {len(processed_tts_text)}, 总时长 {audio_duration_ms}ms, 结尾 {TTS_TAIL_ANALYSIS_DURATION_MS}ms 能量 {current_attempt_tail_dbfs:.2f} dBFS。")

                    if current_attempt_tail_dbfs > TTS_TAIL_ENERGY_THRESHOLD_DBFS:
                        logger.warning(f"  TTS音频结尾能量过高 ({current_attempt_tail_dbfs:.2f} dBFS > {TTS_TAIL_ENERGY_THRESHOLD_DBFS} dBFS)，可能被截断。")
                        
                        # 如果当前尝试是迄今为止“最不戛然而止”的，就更新最佳结果
                        if current_attempt_tail_dbfs < min_tail_dbfs_found_in_retries:
                            min_tail_dbfs_found_in_retries = current_attempt_tail_dbfs
                            best_audio_data_for_saving = current_attempt_audio_data
                            logger.info(f"    更新最佳音频结果 (当前能量: {current_attempt_tail_dbfs:.2f} dBFS)。")

                        await asyncio.sleep(2) # 短暂等待后重试
                        continue # 继续下一次重试
                    else:
                        logger.info(f"  TTS音频结尾能量检查通过。认为正常。")
                        # 成功生成且通过检查，这是最理想的情况，直接保存并跳出
                        best_audio_data_for_saving = current_attempt_audio_data
                        break 

                except Exception as e:
                    logger.error(f"TTS音频结尾能量分析时发生错误: {e} (尝试: {attempt + 1})", exc_info=True)
                    await asyncio.sleep(2) # 短暂等待后重试
                    continue # 继续下一次重试

            except requests.exceptions.RequestException as e:
                logger.error(f"调用 TTS 微服务失败 (尝试: {attempt + 1}): {e}", exc_info=True)
                await asyncio.sleep(2) # 短暂等待后重试
                continue # 继续下一次重试
            except Exception as e:
                logger.error(f"TTS_v2 API 内部处理失败 (尝试: {attempt + 1}): {e}", exc_info=True)
                await asyncio.sleep(2) # 短暂等待后重试
                continue # 继续下一次重试
        
        # 循环结束后，检查是否有可保存的音频数据
        if best_audio_data_for_saving is None:
            logger.error(f"TTS生成失败，所有 {TTS_GENERATION_MAX_RETRIES} 次尝试均未能获取到有效的音频数据。")
            raise HTTPException(status_code=500, detail="TTS生成失败，未能获取到任何音频数据。")

        # 5. 保存音频文件 (保存最佳结果)
        wav_output_dir = os.path.join(OUTPUT_DIR, req.novel_name, 'wavs', req.chapter_name)
        os.makedirs(wav_output_dir, exist_ok=True)
        safe_speaker = "".join(c for c in req.speaker if c.isalnum() or c in " _-").rstrip()
        safe_timbre = "".join(c for c in req.timbre if c.isalnum() or c in " _-").rstrip()
        output_wav_name = f"{req.row_index:04d}-{safe_speaker}-{safe_timbre}.wav"
        output_full_path = os.path.join(wav_output_dir, output_wav_name)
        
        with open(output_full_path, "wb") as f:
            f.write(best_audio_data_for_saving) # 保存最佳数据
        
        return JSONResponse(content={"status": "success", "file_name": output_wav_name})
    
    except requests.exceptions.RequestException as e:
        logger.error(f"调用 TTS 微服务失败: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"无法连接到 TTS 服务: {e}")
    except Exception as e:
        logger.error(f"TTS v2 API 失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

@app.post("/api/splice_audio")
async def splice_audio(req: SpliceRequest):
    # 注意：现在 SpliceRequest 理论上只需要 novel_name 和 chapter_name
    # 但为了兼容性，我们暂时保留 wav_files，但不使用它。

    # --- 1. 构建基础路径 ---
    project_dir = os.path.join(PROJECTS_DIR, req.novel_name)
    timbres_path = os.path.join(project_dir, 'character_timbres.json')
    
    # 使用原始章节名（安全化后）来构建 chapter_json_path 和 wav_input_dir
    # req.chapter_name 此处应为安全化后的文件名，不包含后缀
    safe_chapter_name = req.chapter_name 
    chapter_json_path = os.path.join(project_dir, 'chapters_json', f"{safe_chapter_name}.json")
    wav_input_dir = os.path.join(OUTPUT_DIR, req.novel_name, 'wavs', safe_chapter_name)
    final_output_dir = os.path.join(OUTPUT_DIR, req.novel_name)
    os.makedirs(final_output_dir, exist_ok=True)

    if not os.path.exists(chapter_json_path):
        raise HTTPException(status_code=404, detail=f"章节JSON文件未找到: {chapter_json_path}")

    try:
        # --- 2. 读取配置文件和章节数据 ---
        character_timbres = {}
        if os.path.exists(timbres_path):
            with open(timbres_path, 'r', encoding='utf-8') as f:
                character_timbres = json.load(f)
        
        with open(chapter_json_path, 'r', encoding='utf-8') as f:
            chapter_data = json.load(f)

        # --- 3. 后端自己构建权威的文件列表 ---
        files_to_splice_authoritative = []
        for i, item in enumerate(chapter_data):
            speaker = item.get("speaker")
            
            # 执行与前端完全一致的智能音色判断逻辑
            timbre_to_use = item.get("timbre_override")
            if not timbre_to_use:
                timbre_to_use = character_timbres.get(speaker)

            if not speaker or not timbre_to_use:
                logger.warning(f"跳过拼接第 {i+1} 行，因为缺少角色或音色信息。")
                continue
            
            # 构建正确的文件名
            safe_speaker = "".join(c for c in speaker if c.isalnum() or c in " _-").rstrip()
            safe_timbre = "".join(c for c in timbre_to_use if c.isalnum() or c in " _-").rstrip()
            wav_file_name = f"{i:04d}-{safe_speaker}-{safe_timbre}.wav"
            files_to_splice_authoritative.append(wav_file_name)

        # --- 4. 执行拼接 ---
        if not files_to_splice_authoritative:
            raise HTTPException(status_code=400, detail="根据章节内容，没有找到可拼接的音频文件。")

        combined = AudioSegment.empty()
        for wav_file_name in files_to_splice_authoritative:
            wav_path = os.path.join(wav_input_dir, wav_file_name)
            if os.path.exists(wav_path):
                combined += AudioSegment.from_wav(wav_path)
            else:
                logger.warning(f"拼接时文件未找到，已跳过: {wav_path}")
        
        if len(combined) == 0:
            raise HTTPException(status_code=404, detail="所有预期的WAV文件都不存在。")

        with open(CONFIG_FILE, 'r', encoding='utf-8') as f: config = json.load(f)
        export_settings = config.get("audio_export", {"format": "mp3", "quality": "192k"})
        output_format = export_settings.get("format", "mp3")
        
        output_filename = f"{safe_chapter_name}.{output_format}"
        output_path = os.path.join(final_output_dir, output_filename)
        
        export_params = {}
        output_quality = export_settings.get("quality", "192k") # 先获取质量设置

        if output_format == 'mp3':
            export_params['bitrate'] = output_quality
        elif output_format == 'm4a':
            export_params['codec'] = 'aac'
            export_params['bitrate'] = output_quality
        elif output_format == 'ogg':
            export_params['codec'] = 'libvorbis'
            # quality 参数需要去掉 'q'，例如从 'q5' 变为 '5'
            export_params['parameters'] = ["-q:a", output_quality.replace('q','')]
        
        logger.info(f"正在导出拼接音频到 {output_path}，格式: {output_format}, 参数: {export_params}")
        combined.export(output_path, format=output_format, **export_params)
               
        relative_path = os.path.join(req.novel_name, output_filename).replace("\\", "/")
        return {"status": "success", "file_path": f"/output/{relative_path}"}

    except Exception as e:
        logger.error(f"拼接音频时发生错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器拼接音频失败: {e}")

@app.get("/api/novel/{novel_name}/replace_dict")
async def get_novel_replace_dict(novel_name: str):
    """
    获取指定小说的替换词典规则。
    """
    replace_dict_path = os.path.join(PROJECTS_DIR, novel_name, 'replace_dict.json')
    if not os.path.exists(replace_dict_path):
        return JSONResponse(content={"rules": []})
    
    try:
        with open(replace_dict_path, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
            if not isinstance(rules_data, dict) or "rules" not in rules_data or not isinstance(rules_data["rules"], list):
                logger.warning(f"替换词典文件 '{replace_dict_path}' 结构异常，已重置为空。")
                return JSONResponse(content={"rules": []})
            return JSONResponse(content=rules_data)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"读取或解析替换词典文件 '{replace_dict_path}' 失败: {e}")
        return JSONResponse(content={"rules": []})
    except Exception as e:
        logger.error(f"获取替换词典时发生未知错误: {e}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {e}")

@app.post("/api/novel/{novel_name}/replace_dict")
async def update_novel_replace_dict(novel_name: str, req: UpdateReplaceDictRequest):
    """
    更新（保存）指定小说的替换词典规则。
    """
    project_dir = os.path.join(PROJECTS_DIR, novel_name)
    if not os.path.isdir(project_dir):
        raise HTTPException(status_code=404, detail="小说项目未找到。")
        
    replace_dict_path = os.path.join(project_dir, 'replace_dict.json')
    
    try:
        # 验证每个规则的结构
        for rule in req.rules:
            if not rule.original_word or not rule.replacement_word:
                raise HTTPException(status_code=400, detail="替换规则中的 'original_word' 和 'replacement_word' 不能为空。")

        serializable_rules = [rule.model_dump(mode='json') for rule in req.rules] 
        with open(replace_dict_path, 'w', encoding='utf-8') as f:
            json.dump({"rules": serializable_rules}, f, ensure_ascii=False, indent=4)
        return {"status": "success", "message": f"小说「{novel_name}」的替换词典已保存。"}
    except HTTPException: # 重新抛出自定义的HTTPException
        raise
    except Exception as e:
        logger.error(f"保存小说「{novel_name}」替换词典失败: {e}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {e}")
        
@app.post("/api/update_chapter_content")
async def update_chapter_content(req: UpdateChapterRequest):
    """
    Receives updated chapter content (as a full JSON array) and overwrites the file on the server.
    """
    filepath = req.filepath
    
    # 1. Split the incoming path to construct the correct server path
    try:
        path_parts = filepath.split('/', 1)
        if len(path_parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid filepath format. Expected 'novel_name/chapter.json'.")
        
        novel_name, chapter_filename = path_parts
        
        # Construct the full, correct path on the server
        full_path = os.path.join(PROJECTS_DIR, novel_name, 'chapters_json', chapter_filename)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse filepath.")

    # 2. Security check to prevent directory traversal attacks
    project_root = os.path.abspath(PROJECTS_DIR)
    if not os.path.abspath(full_path).startswith(project_root):
        raise HTTPException(status_code=403, detail="禁止访问。")

    if not os.path.exists(os.path.dirname(full_path)):
        raise HTTPException(status_code=404, detail=f"项目或章节目录未找到: {os.path.dirname(full_path)}")

    # 3. Write the new content to the file
    try:
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(req.content, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Chapter content updated successfully: {full_path}")
        return {"status": "success", "message": "章节内容已成功保存！"}
    except Exception as e:
        logger.error(f"Failed to write to chapter file {full_path}: {e}")
        raise HTTPException(status_code=500, detail=f"保存文件失败: {e}")

# *** 核心修改 2: 添加新的搜索句子API ***
@app.post("/api/search_character_sentences")
async def search_character_sentences(req: SearchSentencesRequest):
    """
    Searches for sentences containing a character's name within the
    original source.txt of selected chapters.
    """
    project_dir = os.path.join(PROJECTS_DIR, req.novel_name)
    source_path = os.path.join(project_dir, 'source.txt')

    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="小说源文件未找到。")

    try:
        with open(source_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
            content = f.read()
        
        all_chapters_map = {c['title']: c['content'] for c in get_chapters_from_txt(content)}
        
        found_sentences = []
        SENTENCE_LIMIT = 20
        
        # 定义一个更智能的句子分割正则表达式，可以处理中英文标点
        sentence_splitter = re.compile(r'([^。！？.…\n]+[。！？.…\n]?)')

        for title in req.chapter_titles:
            if len(found_sentences) >= SENTENCE_LIMIT:
                break
            
            if title in all_chapters_map:
                chapter_content = all_chapters_map[title]
                # 使用正则表达式进行分割
                sentences = sentence_splitter.findall(chapter_content)
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    if req.character_name in sentence:
                        found_sentences.append({
                            "source": title,
                            "content": sentence
                        })
                        if len(found_sentences) >= SENTENCE_LIMIT:
                            break
        
        return {"sentences": found_sentences}

    except Exception as e:
        logger.error(f"搜索角色句子失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器端搜索失败: {e}")
        
def _get_or_create_categories_data():
    """辅助函数：读取分类文件。如果不存在，则根据物理文件夹自动创建。"""
    if os.path.exists(CATEGORIES_FILE):
        with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
            try:
                # 额外增加一个校验，确保基本结构正确
                data = json.load(f)
                if "categories" in data and "unassigned" in data:
                    return data
            except json.JSONDecodeError:
                pass # 文件损坏，将执行下方重建逻辑

    all_timbres = sorted([d for d in os.listdir(WAV_DIR) if os.path.isdir(os.path.join(WAV_DIR, d))])
    data = {"categories": {}, "unassigned": all_timbres}
    with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

def _save_categories_data(data):
    """辅助函数：保存分类数据到文件。"""
    with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@app.get("/api/timbres/data")
async def get_timbres_data():
    """获取所有音色数据，包括分类和未分类列表。"""
    return _get_or_create_categories_data()

@app.post("/api/timbres/categories")
async def create_new_category(req: CreateCategoryRequest): # <-- 使用新的模型
    """创建一个新的空分类。"""
    data = _get_or_create_categories_data()
    new_category_name = req.category_name.strip()
    if not new_category_name:
        raise HTTPException(status_code=400, detail="分类名不能为空。")
    if new_category_name in data["categories"]:
        raise HTTPException(status_code=409, detail="该分类名已存在。")
    
    data["categories"][new_category_name] = []
    _save_categories_data(data)
    return {"status": "success", "message": f"分类 '{new_category_name}' 已创建。", "data": data}

@app.delete("/api/timbres/categories/{category_name}")
async def delete_category(category_name: str):
    """删除一个分类，并将其下的音色移动到“未分类”。"""
    decoded_name = unquote(category_name)
    data = _get_or_create_categories_data()
    if decoded_name in data["categories"]:
        timbres_to_move = data["categories"].pop(decoded_name)
        data["unassigned"].extend(timbres_to_move)
        data["unassigned"] = sorted(list(set(data["unassigned"])))
        _save_categories_data(data)
        return {"status": "success", "message": f"分类 '{decoded_name}' 已删除。", "data": data}
    raise HTTPException(status_code=404, detail="未找到要删除的分类。")
    
@app.post("/api/timbres/move")
async def move_timbre_to_category(req: SetTimbreCategoryRequest):
    """移动一个音色到指定的分类（或未分类）。"""
    data = _get_or_create_categories_data()
    timbre_name = req.timbre_name
    new_category = req.category_name

    # 1. 从所有旧位置移除
    for category, timbres in data["categories"].items():
        if timbre_name in timbres:
            timbres.remove(timbre_name)
            break
    if timbre_name in data["unassigned"]:
        data["unassigned"].remove(timbre_name)
    
    # 2. 添加到新位置
    if not new_category: # 移动到 "未分类"
        data["unassigned"].append(timbre_name)
        data["unassigned"].sort()
    else:
        if new_category not in data["categories"]:
            raise HTTPException(status_code=404, detail="目标分类不存在。")
        data["categories"][new_category].append(timbre_name)
        data["categories"][new_category].sort()

    _save_categories_data(data)
    return {"status": "success", "message": f"音色 '{timbre_name}' 已移动。", "data": data}
    
@app.get("/api/list_timbres")
async def list_timbres():
    """ (兼容接口) 返回所有音色的扁平列表。"""
    data = _get_or_create_categories_data()
    all_timbres_list = list(data['unassigned'])
    for timbres in data['categories'].values():
        all_timbres_list.extend(timbres)
    return {"timbres": sorted(list(set(all_timbres_list)))}

# 【新增】设置音色分类的接口
@app.post("/api/timbres/set_category")
async def set_timbre_category(req: SetTimbreCategoryRequest):
    """为一个音色设置或更改其分类。"""
    data = _get_or_create_categories_data()
    timbre_name = req.timbre_name
    new_category = req.category_name

    # 1. 从所有旧位置（无论是在哪个分类或未分类）中移除该音色
    for category, timbres in data["categories"].items():
        if timbre_name in timbres:
            timbres.remove(timbre_name)
            break
    if timbre_name in data["unassigned"]:
        data["unassigned"].remove(timbre_name)
    
    # 2. 将音色添加到新位置
    if not new_category: # 如果传入空字符串，则移动到“未分类”
        data["unassigned"].append(timbre_name)
        data["unassigned"].sort()
    else: # 否则，移动到指定分类
        # 如果分类不存在，则创建它
        if new_category not in data["categories"]:
            data["categories"][new_category] = []
        
        data["categories"][new_category].append(timbre_name)
        data["categories"][new_category].sort()

    _save_categories_data(data)
    return {"status": "success", "message": f"已将音色 '{timbre_name}' 移动到分类 '{new_category or '未分类'}'。"}

@app.post("/api/upload_timbre")
async def upload_timbre(
    request: Request,
    file: UploadFile = File(...), 
    category_name: str = Form(""),
    timbre_name: str = Form(...), 
    prompt_text: str = Form(...),
    normalize: str = Form(...)
):
    timbre_dir = os.path.join(WAV_DIR, timbre_name)
    if os.path.exists(timbre_dir):
        raise HTTPException(status_code=409, detail="音色名称已存在。")

    # --- 1. 将上传文件保存到临时位置 ---
    temp_input_path = os.path.join(TEMP_DIR, f"input_{uuid.uuid4()}_{file.filename}")
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    current_audio_path = temp_input_path
    
    try:
        # --- 3. 加载音频并进行后续处理 ---
        audio = AudioSegment.from_file(current_audio_path)

        # --- 4. (可选) 音量标准化 ---
        do_normalize = normalize.lower() == 'true'
        if do_normalize:
            logger.info(f"正在为音色 '{timbre_name}' 进行音量标准化...")
            audio = pydub_normalize(audio)
            logger.info("音量标准化完成。")

        # --- 5. 最终格式化并保存 ---
        # 统一转换为单声道
        audio = audio.set_channels(1)
        
        os.makedirs(timbre_dir)
        final_wav_path = os.path.join(timbre_dir, "1.wav")
        audio.export(final_wav_path, format="wav")
        
        with open(os.path.join(timbre_dir, "1.txt"), "w", encoding="utf-8") as f:
            f.write(prompt_text)
        
        data = _get_or_create_categories_data()
        if category_name and category_name in data["categories"]:
            data["categories"][category_name].append(timbre_name)
            data["categories"][category_name].sort()
        else:
            data["unassigned"].append(timbre_name)
            data["unassigned"].sort()
        
        _save_categories_data(data)
        
        return {"status": "success", "message": f"音色 '{timbre_name}' 已成功添加并处理！"}

    except Exception as e:
        logger.error(f"处理上传音色时发生错误: {e}", exc_info=True)
        # 出错时确保清理 timbre_dir
        if os.path.exists(timbre_dir):
            shutil.rmtree(timbre_dir)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # --- 6. 最终清理所有临时文件和目录 ---
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

# *** 删除音色API ***
@app.delete("/api/delete_timbre")
async def delete_timbre(timbre_name: str):
    """
    Deletes a timbre folder and all its contents.
    Includes security checks to prevent directory traversal.
    """
    if not timbre_name:
        raise HTTPException(status_code=400, detail="音色名称不能为空。")
        
    # a. 从 timbre_categories.json 文件中移除音色记录
    data = _get_or_create_categories_data()
    found_and_removed = False
    
    # 从 "未分类" 中查找并移除
    if timbre_name in data["unassigned"]:
        data["unassigned"].remove(timbre_name)
        found_and_removed = True
    
    # 如果未找到，则从所有分类中查找并移除
    if not found_and_removed:
        for category, timbres in data["categories"].items():
            if timbre_name in timbres:
                timbres.remove(timbre_name)
                found_and_removed = True
                break
    
    # 如果找到了并移除了记录，则保存文件
    if found_and_removed:
        _save_categories_data(data)
        logger.info(f"已从 timbre_categories.json 中移除音色记录: {timbre_name}")
    else:
        logger.warning(f"在 timbre_categories.json 中未找到音色记录: {timbre_name}，但仍将尝试删除物理文件夹。")
        
    # b. 删除物理文件夹
    # 1. 构建目标目录的路径
    timbre_dir = os.path.join(WAV_DIR, timbre_name)
    
    # 2. 安全性检查：防止目录遍历攻击
    #    通过 realpath 解析路径，确保它在合法的 WAV_DIR 内部
    wav_root = os.path.realpath(WAV_DIR)
    target_path = os.path.realpath(timbre_dir)
    
    if not target_path.startswith(wav_root):
        logger.warning(f"潜在的目录遍历攻击被阻止: {timbre_name}")
        raise HTTPException(status_code=403, detail="禁止访问。")
        
    if not os.path.isdir(target_path):
        raise HTTPException(status_code=404, detail=f"音色 '{timbre_name}' 未找到。")

    # 3. 执行删除操作
    try:
        shutil.rmtree(target_path)
        logger.info(f"音色已成功删除: {target_path}")
        return {"status": "success", "message": f"音色 '{timbre_name}' 已成功删除。"}
    except Exception as e:
        logger.error(f"删除音色失败 '{timbre_name}': {e}")
        raise HTTPException(status_code=500, detail=f"服务器删除文件失败: {e}")
        
# *** 打包下载API ***
@app.post("/api/download_spliced_chapters")
async def download_spliced_chapters(req: DownloadRequest):
    """
    Takes a list of audio file paths, packages them into a ZIP file in memory,
    and returns it for download.
    """
    if not req.file_paths:
        raise HTTPException(status_code=400, detail="没有提供需要下载的文件路径。")

    zip_buffer = io.BytesIO()
    output_root = os.path.realpath(OUTPUT_DIR)

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in req.file_paths:
            full_path = os.path.join(OUTPUT_DIR, file_path)
            if not os.path.realpath(full_path).startswith(output_root):
                logger.warning(f"下载请求被阻止（目录遍历）: {file_path}")
                continue
            if os.path.exists(full_path):
                zip_file.write(full_path, arcname=os.path.basename(full_path))
            else:
                logger.warning(f"下载请求中包含不存在的文件，已跳过: {full_path}")

    if not zip_file.namelist():
        raise HTTPException(status_code=404, detail="所有请求的文件都不存在或不合法。")

    zip_buffer.seek(0)
    
    novel_name = "chapters"
    if req.file_paths:
        try:
            novel_name = req.file_paths[0].split('/')[0]
        except:
            pass
    
    zip_filename = f"{novel_name}_spliced.zip"
    
    # *** 核心修改：正确处理包含中文的文件名 ***
    try:
        # 尝试将文件名编码为 ASCII，如果失败，说明含有非ASCII字符
        zip_filename.encode('ascii')
        # 如果成功，使用简单的 header
        headers = {
            'Content-Disposition': f'attachment; filename="{zip_filename}"'
        }
    except UnicodeEncodeError:
        # 如果失败，使用 RFC 6266 推荐的格式来处理非ASCII字符
        # 1. 创建一个只包含ASCII字符的回退文件名
        fallback_filename = "download.zip"
        # 2. 对原始文件名进行 URL 编码
        encoded_filename = urllib.parse.quote(zip_filename)
        # 3. 构造复合的 Content-Disposition 头
        headers = {
            'Content-Disposition': f'attachment; filename="{fallback_filename}"; filename*=UTF-8\'\'{encoded_filename}'
        }

    # 使用 StreamingResponse 返回ZIP文件流，并附上我们构造好的headers
    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers=headers
    ) 

@app.post("/api/stt_elevenlabs", response_model=STTResponse)
async def speech_to_text_elevenlabs(file: UploadFile = File(...)):
    # 1. 加载配置，获取 API Key
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
        api_key = config.get("elevenlabs", {}).get("api_key")
        if not api_key:
            raise HTTPException(status_code=503, detail="服务器未配置ElevenLabs API Key。")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="服务器配置文件丢失。")

    # 2. ElevenLabs API 端点和头部
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {
        "xi-api-key": api_key
    }

    # 3. 准备要发送的文件数据
    # a. 定义数据字段 (即使是可选的，也明确发送)
    data = {
        'model_id': 'scribe_v1'
    }
    
    # b. 定义文件字段
    files = {
        'file': (file.filename, await file.read(), file.content_type)
    }
    
    logger.info(f"正在将音频文件 '{file.filename}' 转发到 ElevenLabs STT API...")

    try:
        # 4. 发送请求
        response = requests.post(url, headers=headers, data=data, files=files, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        
        # 5. 解析并返回结果
        recognized_text = result.get("text")
        if recognized_text is not None:
            logger.info("ElevenLabs STT 成功识别文本。")
            return STTResponse(status="success", text=recognized_text)
        else:
            logger.error(f"ElevenLabs API 返回无效响应: {result}")
            raise HTTPException(status_code=500, detail="ElevenLabs API 返回无效响应。")

    except requests.exceptions.RequestException as e:
        logger.error(f"调用 ElevenLabs API 失败: {e}", exc_info=True)
        # 尝试解析ElevenLabs返回的更具体的错误信息
        error_detail = "连接或请求 ElevenLabs API 失败。"
        if e.response is not None:
            try:
                error_data = e.response.json()
                detail_field = error_data.get("detail")

                if isinstance(detail_field, list) and detail_field:
                    # 如果 detail 是一个非空列表
                    first_error = detail_field[0]
                    if isinstance(first_error, dict) and 'msg' in first_error:
                        error_detail = first_error['msg'] # 提取第一个错误的 msg
                elif isinstance(detail_field, dict) and 'message' in detail_field:
                    # 如果 detail 是一个包含 message 的字典
                    error_detail = detail_field['message']
                elif isinstance(detail_field, str):
                    # 如果 detail 本身就是个字符串
                    error_detail = detail_field
                
                # 特别处理 401 Unauthorized 错误
                if e.response.status_code == 401:
                    error_detail = "ElevenLabs API Key 无效或未提供。"

            except json.JSONDecodeError:
                error_detail = f"API返回错误 (状态码: {e.response.status_code})，且响应体不是有效的JSON。"
        
        raise HTTPException(status_code=503, detail=error_detail)
    except Exception as e:
        logger.error(f"STT处理时发生未知错误: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")        
        
# =============================================================
# 5. 静态文件挂载
# =============================================================
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/wav", StaticFiles(directory=WAV_DIR), name="wav")
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# =============================================================
# 6. 主启动逻辑 (核心修复)
# =============================================================
if __name__ == '__main__':

    # 1. 启动时先初始化配置文件
    initialize_llm_config()

    # 2. 配置 FFmpeg
    ffmpeg_executable_path = os.path.join(ROOT_DIR, "ffmpeg-8.0-full_build", "bin", "ffmpeg.exe")
    if os.path.exists(ffmpeg_executable_path):
        AudioSegment.converter = ffmpeg_executable_path
        logger.info(f"成功为 pydub 定位到 ffmpeg.exe: {ffmpeg_executable_path}")
    else:
        logger.warning(f"未在期望的位置找到 ffmpeg.exe: {ffmpeg_executable_path}")
        logger.warning("音频格式转换功能（如导出MP3）可能受限。")

    # 4. 解析命令行参数
    parser = argparse.ArgumentParser(description="AI Voice Studio Pro - Backend Service")
    parser.add_argument('--port', type=int, default=8000, help="端口号")
    parser.add_argument('--host', type=str, default='127.0.0.1', help="主机地址")
    args = parser.parse_args()
    
    logger.info(f"服务器启动，请在浏览器中打开 http://{args.host}:{args.port}")
    
    # 5. 启动 Uvicorn，【不使用】 reload=True
    uvicorn.run(app, host=args.host, port=args.port)