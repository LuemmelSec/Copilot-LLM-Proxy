import json
import logging
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ai-proxy")

CONFIG_PATH = Path(__file__).parent / "config.json"

UNSUPPORTED_FIELDS = {
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "top_logprobs",
}

ANTHROPIC_DROP_FIELDS = UNSUPPORTED_FIELDS | {
    "frequency_penalty",
    "logit_bias",
    "max_completion_tokens",
    "metadata",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "response_format",
    "seed",
    "stop",
    "stream_options",
    "user",
}


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Missing config file: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_config() -> Dict[str, Any]:
    """
    Reload config on every request, so you can edit config.json without
    restarting the proxy.
    """
    return load_config()


def route_config(config: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
    routes = config.get("routes", {}) or {}
    route = routes.get(model or "") or routes.get("default") or {}

    merged = dict(config)
    merged.update(route)
    return merged


def debug_enabled(config: Dict[str, Any]) -> bool:
    debug = config.get("debug", {}) or {}
    return bool(debug.get("enabled", config.get("debug", False)))


def log_debug(config: Dict[str, Any], message: str, **values: Any) -> None:
    if not debug_enabled(config):
        return

    if values:
        logger.info("%s %s", message, json.dumps(values, default=str, ensure_ascii=True))
    else:
        logger.info(message)


def build_headers(config: Dict[str, Any]) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
    }

    headers.update(config.get("extra_headers", {}) or {})

    api_key = config.get("upstream_api_key")
    api_key_env = config.get("upstream_api_key_env")
    if not api_key and api_key_env:
        api_key = os.environ.get(api_key_env)

    auth_header = config.get("auth_header", "Authorization")
    auth_prefix = config.get("auth_prefix", "Bearer")

    if api_key:
        if auth_prefix:
            headers[auth_header] = f"{auth_prefix} {api_key}"
        else:
            headers[auth_header] = api_key

    return headers


def sanitize_body(body: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(body)

    if config.get("remove_unsupported_parameters", True):
        for field in UNSUPPORTED_FIELDS:
            cleaned.pop(field, None)

    force_temperature = config.get("force_temperature")
    if force_temperature is not None:
        cleaned["temperature"] = force_temperature

    model = config.get("upstream_model") or config.get("model")
    if model:
        cleaned["model"] = model

    return cleaned


def openai_models(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    routes = config.get("routes", {}) or {}
    model_ids = [model_id for model_id in routes if model_id != "default"]

    if not model_ids:
        model_ids = [config.get("model", "custom-model")]

    return [
        {
            "id": model_id,
            "object": "model",
            "created": 0,
            "owned_by": "custom",
        }
        for model_id in model_ids
    ]


def upstream_url(config: Dict[str, Any], path: str) -> str:
    base_url = config["upstream_base_url"].rstrip("/")
    return f"{base_url}{path}"


def upstream_query_params(config: Dict[str, Any]) -> Dict[str, str]:
    return config.get("extra_query_params", {}) or {}


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def image_url_to_anthropic_source(url: str) -> Dict[str, Any]:
    if url.startswith("data:") and ";base64," in url:
        header, data = url.split(",", 1)
        media_type = header.removeprefix("data:").split(";", 1)[0]
        return {"type": "base64", "media_type": media_type, "data": data}

    return {"type": "url", "url": url}


def openai_content_to_anthropic(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if not isinstance(content, list):
        return [{"type": "text", "text": coerce_text(content)}]

    blocks: List[Dict[str, Any]] = []
    for item in content:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
            continue

        if not isinstance(item, dict):
            blocks.append({"type": "text", "text": coerce_text(item)})
            continue

        item_type = item.get("type")
        if item_type == "text":
            blocks.append({"type": "text", "text": coerce_text(item.get("text"))})
        elif item_type == "image_url":
            image_url = item.get("image_url", {}) or {}
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if url:
                blocks.append({"type": "image", "source": image_url_to_anthropic_source(url)})
        elif item_type == "input_text":
            blocks.append({"type": "text", "text": coerce_text(item.get("text"))})
        elif item_type == "input_image":
            url = item.get("image_url") or item.get("url")
            if url:
                blocks.append({"type": "image", "source": image_url_to_anthropic_source(url)})

    return blocks or [{"type": "text", "text": ""}]


def openai_tool_call_to_anthropic(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    function = tool_call.get("function", {}) or {}
    arguments = function.get("arguments") or "{}"
    try:
        parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
    except json.JSONDecodeError:
        parsed_arguments = {"arguments": arguments}

    return {
        "type": "tool_use",
        "id": tool_call.get("id") or f"toolu_{uuid4().hex}",
        "name": function.get("name", "tool"),
        "input": parsed_arguments or {},
    }


def openai_tools_to_anthropic(tools: Any) -> List[Dict[str, Any]]:
    anthropic_tools: List[Dict[str, Any]] = []
    if not isinstance(tools, list):
        return anthropic_tools

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        function = tool.get("function", {}) if tool.get("type") == "function" else tool
        name = function.get("name")
        if not name:
            continue

        anthropic_tools.append(
            {
                "name": name,
                "description": function.get("description", ""),
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )

    return anthropic_tools


def openai_tool_choice_to_anthropic(tool_choice: Any) -> Optional[Dict[str, Any]]:
    if tool_choice in (None, "auto"):
        return None
    if tool_choice == "none":
        return {"type": "none"}
    if tool_choice == "required":
        return {"type": "any"}
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function", {}) or {}
        if function.get("name"):
            return {"type": "tool", "name": function["name"]}
    return None


def openai_messages_to_anthropic(messages: Any) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    system_parts: List[str] = []
    anthropic_messages: List[Dict[str, Any]] = []

    if not isinstance(messages, list):
        return None, []

    for message in messages:
        if not isinstance(message, dict):
            continue


        role = message.get("role")
        content = message.get("content")

        if role == "system":
            system_parts.append("\n".join(block.get("text", "") for block in openai_content_to_anthropic(content)))
            continue

        if role == "tool":
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.get("tool_call_id", ""),
                            "content": coerce_text(content),
                        }
                    ],
                }
            )
            continue

        mapped_role = "assistant" if role == "assistant" else "user"
        blocks = openai_content_to_anthropic(content) if content is not None else []

        tool_calls = message.get("tool_calls") or []
        if isinstance(tool_calls, list):
            blocks.extend(openai_tool_call_to_anthropic(tool_call) for tool_call in tool_calls if isinstance(tool_call, dict))

        anthropic_messages.append({"role": mapped_role, "content": blocks or [{"type": "text", "text": ""}]})

    return "\n\n".join(part for part in system_parts if part), anthropic_messages


def openai_chat_to_anthropic(body: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    system, messages = openai_messages_to_anthropic(body.get("messages", []))
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or config.get("default_max_tokens", 4096)

    anthropic_body: Dict[str, Any] = {
        "model": config.get("upstream_model") or body.get("model") or config.get("model"),
        "messages": messages,
        "max_tokens": max_tokens,
    }

    if system:
        anthropic_body["system"] = system

    for field in ("temperature", "top_p", "top_k"):
        if field in body and field not in ANTHROPIC_DROP_FIELDS:
            anthropic_body[field] = body[field]

    stop_sequences = body.get("stop")
    if isinstance(stop_sequences, str):
        anthropic_body["stop_sequences"] = [stop_sequences]
    elif isinstance(stop_sequences, list):
        anthropic_body["stop_sequences"] = stop_sequences

    tools = openai_tools_to_anthropic(body.get("tools"))
    if tools:
        anthropic_body["tools"] = tools

    tool_choice = openai_tool_choice_to_anthropic(body.get("tool_choice"))
    if tool_choice:
        anthropic_body["tool_choice"] = tool_choice

    if body.get("stream"):
        anthropic_body["stream"] = True

    return anthropic_body


def anthropic_stop_to_openai(stop_reason: Optional[str]) -> Optional[str]:
    return {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }.get(stop_reason or "", stop_reason)


def anthropic_message_to_openai_response(message: Dict[str, Any], model: str) -> Dict[str, Any]:
    content_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in message.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            content_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=True),
                    },
                }
            )

    response_message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if tool_calls:
        response_message["tool_calls"] = tool_calls

    usage = message.get("usage", {}) or {}
    return {
        "id": message.get("id") or f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": response_message,
                "finish_reason": anthropic_stop_to_openai(message.get("stop_reason")),
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


def openai_stream_chunk(model: str, delta: Dict[str, Any], finish_reason: Optional[str] = None) -> bytes:
    chunk = {
        "id": f"chatcmpl-{uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=True)}\n\n".encode("utf-8")


async def anthropic_stream_to_openai(upstream_response: httpx.Response, client: httpx.AsyncClient, model: str) -> AsyncIterator[bytes]:
    event_name: Optional[str] = None
    tool_indexes: Dict[int, int] = {}
    next_tool_index = 0
    stop_reason: Optional[str] = None

    yield openai_stream_chunk(model, {"role": "assistant"})

    try:
        async for line in upstream_response.aiter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip()
                continue
            if not line.startswith("data:"):
                continue

            data = line.split(":", 1)[1].strip()
            if not data or data == "[DONE]":
                continue

            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            effective_event = event.get("type") or event_name
            if effective_event == "content_block_start":
                index = int(event.get("index", 0))
                block = event.get("content_block", {}) or {}
                if block.get("type") == "tool_use":
                    tool_index = next_tool_index
                    next_tool_index += 1
                    tool_indexes[index] = tool_index
                    yield openai_stream_chunk(
                        model,
                        {
                            "tool_calls": [
                                {
                                    "index": tool_index,
                                    "id": block.get("id"),
                                    "type": "function",
                                    "function": {"name": block.get("name"), "arguments": ""},
                                }
                            ]
                        },
                    )
            elif effective_event == "content_block_delta":
                index = int(event.get("index", 0))
                delta = event.get("delta", {}) or {}
                if delta.get("type") == "text_delta" and delta.get("text"):
                    yield openai_stream_chunk(model, {"content": delta["text"]})
                elif delta.get("type") == "input_json_delta" and delta.get("partial_json"):
                    tool_index = tool_indexes.get(index, index)
                    yield openai_stream_chunk(
                        model,
                        {"tool_calls": [{"index": tool_index, "function": {"arguments": delta["partial_json"]}}]},
                    )
            elif effective_event == "message_delta":
                delta = event.get("delta", {}) or {}
                stop_reason = delta.get("stop_reason") or stop_reason
            elif effective_event == "message_stop":
                break

        yield openai_stream_chunk(model, {}, anthropic_stop_to_openai(stop_reason) or "stop")
        yield b"data: [DONE]\n\n"
    finally:
        await upstream_response.aclose()
        await client.aclose()


async def forward_openai_chat(body: Dict[str, Any], config: Dict[str, Any]) -> Response:
    body = sanitize_body(body, config)
    url = upstream_url(config, "/chat/completions")
    headers = build_headers(config)
    params = upstream_query_params(config)
    is_stream = bool(body.get("stream"))

    log_debug(config, "forward openai chat", model=body.get("model"), url=url, stream=is_stream)

    if is_stream:
        client = httpx.AsyncClient(timeout=None)

        upstream_request = client.build_request(
            "POST",
            url,
            headers=headers,
            params=params,
            json=body,
        )

        upstream_response = await client.send(upstream_request, stream=True)
        log_debug(config, "upstream openai response", status=upstream_response.status_code, content_type=upstream_response.headers.get("content-type"))

        if upstream_response.status_code >= 400:
            error_body = await upstream_response.aread()
            await upstream_response.aclose()
            await client.aclose()

            return Response(
                content=error_body,
                status_code=upstream_response.status_code,
                media_type=upstream_response.headers.get(
                    "content-type",
                    "application/json",
                ),
            )

        async def stream_generator():
            try:
                async for chunk in upstream_response.aiter_bytes():
                    yield chunk
            finally:
                await upstream_response.aclose()
                await client.aclose()

        return StreamingResponse(
            stream_generator(),
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get(
                "content-type",
                "text/event-stream",
            ),
        )

    async with httpx.AsyncClient(timeout=None) as client:
        upstream_response = await client.post(
            url,
            headers=headers,
            params=params,
            json=body,
        )

    log_debug(config, "upstream openai response", status=upstream_response.status_code, content_type=upstream_response.headers.get("content-type"))

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        media_type=upstream_response.headers.get(
            "content-type",
            "application/json",
        ),
    )


async def forward_anthropic_chat(body: Dict[str, Any], config: Dict[str, Any]) -> Response:
    anthropic_body = openai_chat_to_anthropic(body, config)
    url = upstream_url(config, config.get("messages_path", "/messages"))
    headers = build_headers(config)
    params = upstream_query_params(config)
    is_stream = bool(anthropic_body.get("stream"))
    model = body.get("model") or config.get("model", "anthropic")

    log_debug(config, "forward anthropic chat", model=model, upstream_model=anthropic_body.get("model"), url=url, stream=is_stream)

    if is_stream:
        client = httpx.AsyncClient(timeout=None)
        upstream_request = client.build_request("POST", url, headers=headers, params=params, json=anthropic_body)
        upstream_response = await client.send(upstream_request, stream=True)
        log_debug(config, "upstream anthropic response", status=upstream_response.status_code, content_type=upstream_response.headers.get("content-type"))

        if upstream_response.status_code >= 400:
            error_body = await upstream_response.aread()
            await upstream_response.aclose()
            await client.aclose()
            return Response(
                content=error_body,
                status_code=upstream_response.status_code,
                media_type=upstream_response.headers.get("content-type", "application/json"),
            )

        return StreamingResponse(
            anthropic_stream_to_openai(upstream_response, client, model),
            status_code=upstream_response.status_code,
            media_type="text/event-stream",
        )

    async with httpx.AsyncClient(timeout=None) as client:
        upstream_response = await client.post(url, headers=headers, params=params, json=anthropic_body)

    log_debug(config, "upstream anthropic response", status=upstream_response.status_code, content_type=upstream_response.headers.get("content-type"))

    if upstream_response.status_code >= 400:
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type=upstream_response.headers.get("content-type", "application/json"),
        )

    return JSONResponse(anthropic_message_to_openai_response(upstream_response.json(), model))


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "Copilot Python proxy is running",
        "openai_compatible_base_url": "http://127.0.0.1:8787/v1"
    }


@app.get("/v1/models")
async def models():
    config = get_config()

    return JSONResponse(
        {
            "object": "list",
            "data": openai_models(config),
        }
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    config = get_config()

    body = await request.json()
    route = route_config(config, body.get("model"))
    provider = route.get("provider", "openai")

    if provider == "anthropic":
        return await forward_anthropic_chat(body, route)

    return await forward_openai_chat(body, route)


@app.post("/v1/responses")
async def responses(request: Request):
    """
    Optional support in case the client uses OpenAI's newer Responses API.
    """
    config = get_config()

    body = await request.json()
    body = sanitize_body(body, config)

    url = upstream_url(config, "/responses")
    headers = build_headers(config)
    params = upstream_query_params(config)

    async with httpx.AsyncClient(timeout=None) as client:
        upstream_response = await client.post(
            url,
            headers=headers,
            params=params,
            json=body,
        )

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        media_type=upstream_response.headers.get(
            "content-type",
            "application/json",
        ),
    )
