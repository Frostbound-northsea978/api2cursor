"""路由层公共辅助

收敛多个数据面路由都会用到的上下文解析、上游目标构造、日志输出和
SSE 消息拼装逻辑，避免 `chat.py` 和 `responses.py` 各自维护重复实现。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import threading
import time
from typing import Any

import settings
from utils.http import build_anthropic_headers, build_gemini_headers, build_openai_headers

logger = logging.getLogger(__name__)

_RESPONSES_PREV_ID_LOCK = threading.Lock()
_RESPONSES_PREV_ID_TTL = 86400
_RESPONSES_PREV_IDS: dict[str, tuple[str, float]] = {}


@dataclass(frozen=True)
class RouteContext:
    """数据面路由使用的标准请求上下文。

    路由层会先根据客户端模型名解析出统一上下文，后续处理函数只需要关心
    上游模型、后端类型、目标地址、鉴权信息、流式标记和自定义指令，
    而不必重复访问配置层。
    """

    client_model: str
    upstream_model: str
    backend: str
    target_url: str
    api_key: str
    is_stream: bool
    custom_instructions: str
    instructions_position: str
    body_modifications: dict
    header_modifications: dict


def build_route_context(client_model: str, is_stream: bool) -> RouteContext:
    """解析模型映射，得到当前请求的统一路由上下文。"""
    mapping = settings.resolve_model(client_model)
    return RouteContext(
        client_model=client_model,
        upstream_model=mapping['upstream_model'],
        backend=mapping['backend'],
        target_url=mapping['target_url'],
        api_key=mapping['api_key'],
        is_stream=is_stream,
        custom_instructions=mapping.get('custom_instructions', ''),
        instructions_position=mapping.get('instructions_position', 'prepend'),
        body_modifications=mapping.get('body_modifications', {}),
        header_modifications=mapping.get('header_modifications', {}),
    )


def build_openai_target(ctx: RouteContext) -> tuple[str, dict[str, str]]:
    """根据路由上下文生成 OpenAI 兼容后端的地址和请求头。"""
    url = f'{ctx.target_url.rstrip("/")}/v1/chat/completions'
    headers = build_openai_headers(ctx.api_key)
    return url, headers


def build_responses_target(ctx: RouteContext) -> tuple[str, dict[str, str]]:
    """根据路由上下文生成 OpenAI Responses 后端的地址和请求头。"""
    url = f'{ctx.target_url.rstrip("/")}/v1/responses'
    headers = build_openai_headers(ctx.api_key)
    return url, headers


def build_anthropic_target(ctx: RouteContext) -> tuple[str, dict[str, str]]:
    """根据路由上下文生成 Anthropic 后端的地址和请求头。"""
    url = f'{ctx.target_url.rstrip("/")}/v1/messages'
    headers = build_anthropic_headers(ctx.api_key)
    return url, headers


def build_gemini_target(ctx: RouteContext, stream: bool = False) -> tuple[str, dict[str, str]]:
    """根据路由上下文生成 Gemini 后端的地址和请求头。

    Gemini URL 格式: {base}/v1/models/{model}:generateContent
    流式: {base}/v1/models/{model}:streamGenerateContent?alt=sse
    """
    base = ctx.target_url.rstrip('/')
    model = ctx.upstream_model
    if stream:
        url = f'{base}/v1/models/{model}:streamGenerateContent?alt=sse'
    else:
        url = f'{base}/v1/models/{model}:generateContent'
    headers = build_gemini_headers(ctx.api_key)
    return url, headers


def log_route_context(route_name: str, ctx: RouteContext, *, extra: str = '') -> None:
    """统一输出路由级日志，避免不同入口的日志格式逐渐漂移。"""
    parts = [
        f'[{route_name}]',
        f'模型={ctx.client_model}',
        f'上游模型={ctx.upstream_model}',
        f'后端={ctx.backend}',
        f'流式={ctx.is_stream}',
    ]
    if extra:
        parts.append(extra)
    logger.info(' '.join(parts))


def log_usage(
    route_name: str,
    usage: dict[str, Any],
    *,
    input_key: str,
    output_key: str,
) -> None:
    """统一输出令牌统计日志。

    不同协议对 usage 字段命名不一致，这里只接收字段名，不在调用方重复拼接日志文案。
    """
    logger.info(
        '[%s] 请求完成 输入令牌=%s 输出令牌=%s',
        route_name,
        usage.get(input_key, 0),
        usage.get(output_key, 0),
    )


def sse_data_message(data: Any) -> str:
    """构造仅包含 data 的 SSE 消息。"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f'data: {payload}\n\n'


def sse_event_message(event_type: str, data: Any) -> str:
    """构造带 event 名称的 SSE 消息。"""
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f'event: {event_type}\ndata: {payload}\n\n'


def chat_error_chunk(message: str, error_type: str = 'upstream_error') -> str:
    """构造聊天补全流式接口使用的错误消息。"""
    return sse_data_message({'error': {'message': message, 'type': error_type}})


def responses_error_event(message: str) -> str:
    """构造 Responses 流式接口使用的错误事件。"""
    return sse_event_message('error', {'error': message})


# ─── 自定义指令注入 ──────────────────────────────


def _merge_text(custom: str, existing: str, position: str) -> str:
    """根据 position 决定自定义指令与原有内容的拼接顺序。"""
    if not existing:
        return custom
    if position == 'append':
        return existing + '\n\n' + custom
    return custom + '\n\n' + existing


def inject_instructions_cc(payload: dict[str, Any], instructions: str, position: str = 'prepend') -> dict[str, Any]:
    """向 Chat Completions 请求注入自定义指令。

    position='prepend' 时放在 system 消息开头，'append' 时放在末尾。
    """
    if not instructions:
        return payload

    messages = payload.get('messages', [])
    if messages and messages[0].get('role') == 'system':
        first = messages[0]
        original = first.get('content') or ''
        first['content'] = _merge_text(instructions, original, position)
    else:
        messages.insert(0, {'role': 'system', 'content': instructions})
        payload['messages'] = messages

    logger.info('已注入自定义指令到 CC system 消息 (%d 字符, %s)', len(instructions), position)
    return payload


def inject_instructions_responses(payload: dict[str, Any], instructions: str, position: str = 'prepend') -> dict[str, Any]:
    """向 Responses 请求注入自定义指令（写入 instructions 字段）。

    position='prepend' 时放在 instructions 开头，'append' 时放在末尾。
    """
    if not instructions:
        return payload

    existing = payload.get('instructions') or ''
    payload['instructions'] = _merge_text(instructions, existing, position)

    logger.info('已注入自定义指令到 Responses instructions (%d 字符, %s)', len(instructions), position)
    return payload


def ensure_responses_cache_control(payload: dict[str, Any]) -> dict[str, Any]:
    """为 Responses 请求补齐自动 prompt caching 开关。

    一些支持 `/v1/responses` 的上游会参考顶层 `cache_control` 来自动放置缓存断点。
    Cursor 侧通常不会主动携带这个字段，因此这里在缺失时补一个保守的默认值，
    同时允许调用方通过 body_modifications 或显式字段自行覆盖/关闭。
    """
    if not isinstance(payload, dict):
        return payload
    cache_control = payload.get('cache_control')
    if isinstance(cache_control, dict) and cache_control.get('type'):
        return payload
    payload['cache_control'] = {'type': 'ephemeral'}
    logger.info('已为 Responses 请求自动启用 cache_control=ephemeral')
    return payload


def attach_previous_response_id(payload: dict[str, Any]) -> dict[str, Any]:
    """为多轮 Responses 请求补齐上一轮 response_id。

    某些上游在 `/v1/responses` 多轮场景下，只有沿用 `previous_response_id` 才能稳定复用
    上一轮的服务端响应链与缓存。Cursor 通常会回传完整历史，但不会主动带这个字段，
    因此代理需要基于稳定对话键做一次轻量补齐。
    """
    if not isinstance(payload, dict) or payload.get('previous_response_id'):
        return payload
    key = _responses_prev_id_key(payload)
    if not key:
        return payload
    previous_response_id = _get_previous_response_id(key)
    if not previous_response_id:
        return payload
    payload['previous_response_id'] = previous_response_id
    logger.info('已为 Responses 请求补齐 previous_response_id')
    return payload


def remember_response_id(payload: dict[str, Any], response_data: dict[str, Any]) -> None:
    """记住当前对话最近一次上游 Responses response_id。"""
    if not isinstance(payload, dict) or not isinstance(response_data, dict):
        return
    response_id = response_data.get('id')
    if not isinstance(response_id, str) or not response_id.strip():
        return
    key = _responses_prev_id_key(payload)
    if not key:
        return
    with _RESPONSES_PREV_ID_LOCK:
        _RESPONSES_PREV_IDS[key] = (response_id.strip(), time.time())
        _cleanup_previous_response_ids_locked()


def _responses_prev_id_key(payload: dict[str, Any]) -> str:
    """基于 Responses 请求的“对话根信息”生成稳定键。

    这里故意不直接使用完整 `input` 作为键，因为多轮对话每轮都会追加历史；
    如果把整段历史都纳入哈希，键会在每一轮变化，导致无法稳定取回上一轮的
    `previous_response_id`。当前策略只取 instructions 与首轮 user/assistant 根消息。
    """
    instructions = payload.get('instructions') or ''
    input_data = payload.get('input', [])
    if isinstance(input_data, str):
        seed_input = input_data
    elif isinstance(input_data, list):
        seed_input = _responses_root_seed_from_items(input_data)
    else:
        seed_input = json.dumps(input_data, ensure_ascii=False, default=str)
    raw = instructions + '|' + seed_input
    if not raw.strip('|'):
        return ''
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]


def _responses_root_seed_from_items(items: list[Any]) -> str:
    """从 Responses `input` 中提取足够稳定的对话根片段。

    目标不是完整还原会话，而是构造一个在同一段对话内尽量恒定、跨轮次可复用的
    seed。这里沿用项目里 conversation seed 的思路：优先取第一条 user 与第一条
    assistant；如果 assistant 还不存在，则只用第一条 user。
    """
    first_user = None
    first_assistant = None
    for item in items:
        if isinstance(item, str):
            if first_user is None:
                first_user = {'role': 'user', 'content': item}
            continue
        if not isinstance(item, dict):
            continue
        item_type = item.get('type', '')
        role = item.get('role', '')
        if item_type == 'message' and role in ('user', 'assistant'):
            normalized = {
                'role': role,
                'content': _responses_normalize_content(item.get('content', [])),
            }
            if role == 'user' and first_user is None:
                first_user = normalized
            elif role == 'assistant' and first_assistant is None:
                first_assistant = normalized
        elif role in ('user', 'assistant') and not item_type:
            normalized = {
                'role': role,
                'content': _responses_normalize_content(item.get('content', '')),
            }
            if role == 'user' and first_user is None:
                first_user = normalized
            elif role == 'assistant' and first_assistant is None:
                first_assistant = normalized
        if first_user is not None and first_assistant is not None:
            break
    parts = []
    if first_user is not None:
        parts.append(first_user)
    if first_assistant is not None:
        parts.append(first_assistant)
    return json.dumps(parts, ensure_ascii=False, separators=(',', ':'))


def _responses_normalize_content(content: Any) -> str:
    """把 Responses 各种 content 形态折叠成稳定文本。

    这里的目标不是保真展示，而是降低结构差异对 key 计算的影响；只抽取会影响
    会话根语义的文本型内容，忽略无关字段，避免同一轮请求因格式细节不同而得到
    不同的 previous_response_id 键。
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return str(content).strip() if content is not None else ''
    texts: list[str] = []
    for part in content:
        if isinstance(part, str):
            texts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        if part.get('type') in ('input_text', 'output_text', 'text'):
            texts.append(part.get('text', ''))
        elif part.get('type') == 'summary_text':
            texts.append(part.get('text', ''))
    return '\n'.join(texts).strip()


def _get_previous_response_id(key: str) -> str:
    """按稳定键读取上一轮 response_id，并在过期时顺手清理。"""
    with _RESPONSES_PREV_ID_LOCK:
        entry = _RESPONSES_PREV_IDS.get(key)
        if not entry:
            return ''
        response_id, ts = entry
        if (time.time() - ts) >= _RESPONSES_PREV_ID_TTL:
            _RESPONSES_PREV_IDS.pop(key, None)
            return ''
        return response_id


def _cleanup_previous_response_ids_locked() -> None:
    """清理过期的 previous_response_id 缓存项。

    这张表只用于短期多轮续接；一旦对话长时间不活跃，就不再需要继续保留，
    以免常驻进程运行过久后累计过多失效状态。
    """
    now = time.time()
    expired = [
        key for key, (_, ts) in _RESPONSES_PREV_IDS.items()
        if (now - ts) >= _RESPONSES_PREV_ID_TTL
    ]
    for key in expired:
        _RESPONSES_PREV_IDS.pop(key, None)


def inject_instructions_anthropic(payload: dict[str, Any], instructions: str, position: str = 'prepend') -> dict[str, Any]:
    """向 Anthropic Messages 请求注入自定义指令（写入 system 字段）。

    position='prepend' 时放在 system 开头，'append' 时放在末尾。
    """
    if not instructions:
        return payload

    existing = payload.get('system') or ''
    if isinstance(existing, list):
        existing = '\n'.join(
            block.get('text', '') for block in existing
            if isinstance(block, dict) and block.get('type') == 'text'
        )
    payload['system'] = _merge_text(instructions, existing, position)

    logger.info('已注入自定义指令到 Anthropic system (%d 字符, %s)', len(instructions), position)
    return payload


# ─── Body / Header 修改 ──────────────────────────


def apply_body_modifications(payload: dict[str, Any], modifications: dict[str, Any]) -> dict[str, Any]:
    """对转发请求体应用字段级修改。

    规则与 CursorProxy 一致：值为 null 的字段会被删除，其余字段设置/覆盖。
    """
    if not modifications:
        return payload
    for key, value in modifications.items():
        if value is None:
            payload.pop(key, None)
        else:
            payload[key] = value
    logger.info('已应用 body_modifications: %s', list(modifications.keys()))
    return payload


def apply_header_modifications(headers: dict[str, str], modifications: dict[str, Any]) -> dict[str, str]:
    """对转发请求头应用字段级修改。

    规则同 body：值为 null 删除，其余设置/覆盖。
    """
    if not modifications:
        return headers
    for key, value in modifications.items():
        if value is None:
            headers.pop(key, None)
        else:
            headers[key] = str(value)
    logger.info('已应用 header_modifications: %s', list(modifications.keys()))
    return headers
