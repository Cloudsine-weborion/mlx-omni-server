import time
import uuid
import base64
import mimetypes
import os
import re
import subprocess
import tempfile
from typing import Generator, Any, Dict, List, Optional, Tuple

from mlx_omni_server.chat.mlx.chat_generator import DEFAULT_MAX_TOKENS, ChatGenerator
from mlx_omni_server.chat.openai.schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
    Role,
)
from mlx_omni_server.utils.logger import logger


# ---> OpenAI router create_chat_completion > [vision_preprocess_messages] > augment messages with image understanding via mlx-vlm
def _is_data_url(value: str) -> bool:
    return value.startswith("data:")


def _write_data_url_to_temp_file(data_url: str) -> str:
    match = re.match(r"^data:([^;]+);base64,(.*)$", data_url)
    if not match:
        raise ValueError("Unsupported data URL format for image")
    mime_type, b64_data = match.group(1), match.group(2)
    ext = mimetypes.guess_extension(mime_type) or ".bin"
    raw = base64.b64decode(b64_data)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp.write(raw)
        tmp.flush()
    finally:
        tmp.close()
    return tmp.name


def _download_to_temp_file(url: str) -> str:
    import requests  # lazy import

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "application/octet-stream")
    ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        tmp.write(resp.content)
        tmp.flush()
    finally:
        tmp.close()
    return tmp.name


def _resolve_image_ref_to_path(image_ref: str) -> Tuple[str, Optional[str]]:
    """Return (path, temp_path_to_cleanup_or_None)."""
    if _is_data_url(image_ref):
        path = _write_data_url_to_temp_file(image_ref)
        return path, path
    if image_ref.startswith("http://") or image_ref.startswith("https://"):
        path = _download_to_temp_file(image_ref)
        return path, path
    # Local path
    if os.path.exists(image_ref):
        return image_ref, None
    raise FileNotFoundError(f"Image not found or unsupported reference: {image_ref}")


def _extract_image_refs_from_message_content(content: Any) -> Tuple[str, List[str]]:
    """Return (text, image_refs[]) from an OpenAI-style content field."""
    if isinstance(content, str):
        return content, []

    text_parts: List[str] = []
    image_refs: List[str] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text" and "text" in item:
                text_parts.append(str(item.get("text", "")))
            elif item_type in ("image", "image_url"):
                image_url = item.get("image_url")
                # Support both string and {url: ...}
                if isinstance(image_url, dict):
                    url_value = image_url.get("url")
                    if url_value:
                        image_refs.append(url_value)
                elif isinstance(image_url, str):
                    image_refs.append(image_url)
    return ("\n\n".join([p for p in text_parts if p.strip()]), image_refs)


def _extract_generation_text_from_mlx_vlm_output(output: str) -> str:
    """Return only the model's generated text from mlx-vlm CLI stdout.

    The CLI prints progress, prompts, and metrics. We heuristically keep the
    longest contiguous block of lines that are not known metadata prefixes.
    """
    if not output:
        return ""

    lines = [ln.strip() for ln in output.splitlines()]
    skip_prefixes = (
        "Fetching ",
        "Files:",
        "Prompt:",
        "Generation:",
        "Peak memory:",
        "Using ",
        "Calling `python -m mlx_vlm.generate",
        "WARNING",
        "INFO",
        "DEBUG",
        "==========",
    )

    segments: List[List[str]] = []
    current: List[str] = []
    for ln in lines:
        if not ln or any(ln.startswith(pfx) for pfx in skip_prefixes):
            if current:
                segments.append(current)
                current = []
            continue
        current.append(ln)
    if current:
        segments.append(current)

    if not segments:
        return ""

    # Choose the longest text segment assuming it's the actual generation.
    best = max(segments, key=lambda seg: sum(len(s) for s in seg))
    return "\n".join(best).strip()


# ---> OpenAIAdapter.generate/generate_stream > [_run_mlx_vlm] > direct caption extraction for vision inputs
def _run_mlx_vlm(prompt: str, image_path: str, model_name: Optional[str] = None) -> str:
    """Call mlx-vlm CLI and return ONLY the generated text."""
    chosen_model = (
        model_name
        or os.environ.get("MLX_VLM_MODEL")
        or "mlx-community/gemma-3-12b-it-4bit"
    )

    cmd = [
        "python",
        "-m",
        "mlx_vlm.generate",
        "--model",
        chosen_model,
        "--max-tokens",
        "100",
        "--temperature",
        "0.0",
        "--prompt",
        prompt,
        "--image",
        image_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=120)
        return _extract_generation_text_from_mlx_vlm_output(out)
    except Exception as e:
        logger.error(f"mlx-vlm invocation failed: {e}")
        return ""


def _augment_messages_with_image_understanding(messages: List[Dict[str, Any]], model_name: Optional[str]) -> List[Dict[str, Any]]:
    augmented: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        text, image_refs = _extract_image_refs_from_message_content(content)
        if not image_refs:
            augmented.append(msg)
            continue

        prompt = text if text and text.strip() else "Describe this image."
        captions: List[str] = []
        temp_files: List[str] = []
        try:
            for idx, ref in enumerate(image_refs, start=1):
                try:
                    path, tmp = _resolve_image_ref_to_path(ref)
                    if tmp:
                        temp_files.append(tmp)
                    caption = _run_mlx_vlm(prompt, path, model_name)
                    if caption:
                        if len(image_refs) > 1:
                            captions.append(f"Image {idx}: {caption}")
                        else:
                            captions.append(caption)
                except Exception as e:
                    logger.error(f"Failed to process image ref: {e}")
                    continue
        finally:
            # Cleanup temp files
            for f in temp_files:
                try:
                    os.unlink(f)
                except Exception:
                    pass

        combined_text = prompt
        if captions:
            combined_text = (prompt + "\n\n" + "\n\n".join(captions)).strip()

        new_msg = dict(msg)
        new_msg["content"] = combined_text
        # Remove tool_calls or other fields untouched
        augmented.append(new_msg)

    return augmented


# ---> OpenAI router create_chat_completion > [_gather_vlm_captions_from_request] > short-circuit response for vision inputs
def _gather_vlm_captions_from_request(
    request: ChatCompletionRequest,
    model_name: Optional[str],
) -> Tuple[bool, str]:
    """Collect captions for any image content in the request; returns (has_images, text)."""
    captions: List[str] = []
    has_images = False
    temp_files: List[str] = []
    try:
        for msg in request.messages:
            text, image_refs = _extract_image_refs_from_message_content(msg.content)
            if not image_refs:
                continue
            has_images = True
            prompt = text if text and text.strip() else "Describe this image."
            for idx, ref in enumerate(image_refs, start=1):
                try:
                    path, tmp = _resolve_image_ref_to_path(ref)
                    if tmp:
                        temp_files.append(tmp)
                    caption = _run_mlx_vlm(prompt, path, model_name)
                    if caption:
                        if len(image_refs) > 1:
                            captions.append(f"Image {idx}: {caption}")
                        else:
                            captions.append(caption)
                except Exception as e:
                    logger.error(f"Failed to process image ref: {e}")
                    continue
    finally:
        for f in temp_files:
            try:
                os.unlink(f)
            except Exception:
                pass

    return has_images, "\n\n".join(captions).strip()


class OpenAIAdapter:
    """MLX Chat Model wrapper with internal parameter management"""

    def __init__(
        self,
        wrapper: ChatGenerator,
    ):
        """Initialize MLXModel with wrapper object.

        Args:
            wrapper: ChatGenerator instance (cached and ready to use)
        """
        self._default_max_tokens = DEFAULT_MAX_TOKENS
        self._generate_wrapper = wrapper

    # ---> OpenAIAdapter.generate/generate_stream > [_prepare_generation_params] > ChatGenerator.generate/generate_stream
    def _prepare_generation_params(self, request: ChatCompletionRequest) -> dict:
        """Prepare common parameters for both generate and stream_generate."""
        max_tokens = (
            request.max_completion_tokens
            or request.max_tokens
            or self._default_max_tokens
        )

        # Extract parameters from request and extra params
        extra_params = request.get_extra_params()
        extra_body = extra_params.get("extra_body", {})

        # Prepare sampler configuration
        sampler_config = {
            "temp": 1.0 if request.temperature is None else request.temperature,
            "top_p": 1.0 if request.top_p is None else request.top_p,
            "top_k": extra_body.get("top_k", 0),
        }

        # Add additional sampler parameters from extra_body
        if extra_body.get("min_p") is not None:
            sampler_config["min_p"] = extra_body.get("min_p")
        if extra_body.get("min_tokens_to_keep") is not None:
            sampler_config["min_tokens_to_keep"] = extra_body.get("min_tokens_to_keep")
        if extra_body.get("xtc_probability") is not None:
            sampler_config["xtc_probability"] = extra_body.get("xtc_probability")
        if extra_body.get("xtc_threshold") is not None:
            sampler_config["xtc_threshold"] = extra_body.get("xtc_threshold")

        # Prepare template parameters - include both extra_body and direct extra params
        template_kwargs = dict(extra_body)

        # Handle direct extra parameters (for backward compatibility)
        for key in ["enable_thinking"]:
            if key in extra_params:
                template_kwargs[key] = extra_params[key]

        # Convert messages to dict format
        messages = [
            {
                "role": (
                    msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                ),
                "content": msg.content,
                **({"name": msg.name} if msg.name else {}),
                **({"tool_calls": msg.tool_calls} if msg.tool_calls else {}),
            }
            for msg in request.messages
        ]

        # Vision augmentation: if not short-circuiting, fold mlx-vlm captions
        # into text so downstream text-only tokenizer can reason over images.
        try:
            vlm_model_name = (
                extra_body.get("vlm_model")
                or extra_body.get("vision_model")
                or request.model
            )
            messages = _augment_messages_with_image_understanding(messages, vlm_model_name)
        except Exception as e:
            logger.error(f"Vision preprocessing failed, continuing without image context: {e}")

        # Convert tools to dict format
        tools = None
        if request.tools:
            tools = [
                tool.model_dump() if hasattr(tool, "model_dump") else dict(tool)
                for tool in request.tools
            ]

        logger.info(f"messages: {messages}")
        logger.info(f"template_kwargs: {template_kwargs}")

        json_schema = None
        if request.response_format and request.response_format.json_schema:
            json_schema = request.response_format.json_schema.schema_def

        return {
            "messages": messages,
            "tools": tools,
            "max_tokens": max_tokens,
            "sampler": sampler_config,
            "top_logprobs": request.top_logprobs if request.logprobs else None,
            "template_kwargs": template_kwargs,
            "enable_prompt_cache": False,
            "repetition_penalty": request.presence_penalty,
            "json_schema": json_schema,
        }

    def generate(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        # ---> OpenAI router create_chat_completion > [generate] > response body: /v1/chat/completions
        """Generate complete response using the wrapper."""
        try:
            # Short-circuit for vision: return only mlx-vlm output when images present
            extra_params = request.get_extra_params()
            extra_body = extra_params.get("extra_body", {})
            vlm_model_name = (
                extra_body.get("vlm_model")
                or extra_body.get("vision_model")
                or request.model
            )
            has_images, vlm_text = _gather_vlm_captions_from_request(request, vlm_model_name)
            if has_images and vlm_text:
                logger.debug("Returning VLM generation directly without text model")
                message = ChatMessage(role=Role.ASSISTANT, content=vlm_text)
                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=message,
                            finish_reason="stop",
                            logprobs=None,
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        prompt_tokens_details=None,
                    ),
                )

            # Prepare parameters
            params = self._prepare_generation_params(request)

            # Directly use wrapper's generate method for complete response
            result = self._generate_wrapper.generate(**params)

            logger.debug(f"Model Response:\n{result.content.text}")

            # Use reasoning from the wrapper's result
            final_content = result.content.text
            reasoning_content = result.content.reasoning

            # Use wrapper's chat tokenizer for tool processing
            if request.tools:
                message = ChatMessage(
                    role=Role.ASSISTANT,
                    content=final_content,
                    tool_calls=result.content.tool_calls,
                    reasoning=reasoning_content,
                )
            else:
                message = ChatMessage(
                    role=Role.ASSISTANT,
                    content=final_content,
                    reasoning=reasoning_content,
                )

            # Use cached tokens from wrapper stats
            cached_tokens = result.stats.cache_hit_tokens
            logger.debug(f"Generate response with {cached_tokens} cached tokens")

            prompt_tokens_details = None
            if cached_tokens > 0:
                from .schema import PromptTokensDetails

                prompt_tokens_details = PromptTokensDetails(cached_tokens=cached_tokens)

            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:10]}",
                created=int(time.time()),
                model=request.model,
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=message,
                        finish_reason=(
                            "tool_calls"
                            if message.tool_calls
                            else (result.finish_reason or "stop")
                        ),
                        logprobs=result.logprobs,
                    )
                ],
                usage=ChatCompletionUsage(
                    prompt_tokens=result.stats.prompt_tokens + cached_tokens,
                    completion_tokens=result.stats.completion_tokens,
                    total_tokens=result.stats.prompt_tokens
                    + result.stats.completion_tokens
                    + cached_tokens,
                    prompt_tokens_details=prompt_tokens_details,
                ),
            )
        except Exception as e:
            logger.error(f"Failed to generate completion: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to generate completion: {str(e)}")

    def generate_stream(
        self,
        request: ChatCompletionRequest,
    ) -> Generator[ChatCompletionChunk, None, None]:
        # ---> OpenAI router create_chat_completion(stream) > [generate_stream] > SSE stream: /v1/chat/completions
        """Stream generate OpenAI-compatible chunks."""
        try:
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:10]}"

            # Prepare parameters
            # Short-circuit for vision: stream only mlx-vlm output when images present
            extra_params = request.get_extra_params()
            extra_body = extra_params.get("extra_body", {})
            vlm_model_name = (
                extra_body.get("vlm_model")
                or extra_body.get("vision_model")
                or request.model
            )
            has_images, vlm_text = _gather_vlm_captions_from_request(request, vlm_model_name)
            if has_images and vlm_text:
                created = int(time.time())
                # Emit a single delta with full text, then stop
                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatMessage(role=Role.ASSISTANT, content=vlm_text),
                            finish_reason=None,
                            logprobs=None,
                        )
                    ],
                )
                created = int(time.time())
                include_usage = bool(
                    request.stream_options and request.stream_options.include_usage
                )
                final_kwargs = {}
                if include_usage:
                    final_kwargs["usage"] = ChatCompletionUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        prompt_tokens_details=None,
                    )

                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=ChatMessage(role=Role.ASSISTANT),
                            finish_reason="stop",
                            logprobs=None,
                        )
                    ],
                    **final_kwargs,
                )
                return

            params = self._prepare_generation_params(request)

            result = None
            for chunk in self._generate_wrapper.generate_stream(**params):
                created = int(time.time())

                # TODO: support streaming tools parse
                # For streaming, we need to get the appropriate delta content
                if chunk.content.text_delta:
                    content = chunk.content.text_delta
                elif chunk.content.reasoning_delta:
                    content = chunk.content.reasoning_delta
                else:
                    content = ""

                message = ChatMessage(role=Role.ASSISTANT, content=content)

                yield ChatCompletionChunk(
                    id=chat_id,
                    created=created,
                    model=request.model,
                    choices=[
                        ChatCompletionChunkChoice(
                            index=0,
                            delta=message,
                            finish_reason=chunk.finish_reason,
                            logprobs=chunk.logprobs,
                        )
                    ],
                )
                result = chunk

            # Emit a final stop chunk; include usage if requested
            created = int(time.time())
            include_usage = bool(
                request.stream_options and request.stream_options.include_usage and result is not None
            )

            final_kwargs = {}
            if include_usage:
                cached_tokens = result.stats.cache_hit_tokens
                logger.debug(f"Stream response with {cached_tokens} cached tokens")

                prompt_tokens_details = None
                if cached_tokens > 0:
                    from .schema import PromptTokensDetails

                    prompt_tokens_details = PromptTokensDetails(
                        cached_tokens=cached_tokens
                    )

                final_kwargs["usage"] = ChatCompletionUsage(
                    prompt_tokens=result.stats.prompt_tokens + cached_tokens,
                    completion_tokens=result.stats.completion_tokens,
                    total_tokens=result.stats.prompt_tokens
                    + result.stats.completion_tokens
                    + cached_tokens,
                    prompt_tokens_details=prompt_tokens_details,
                )

            yield ChatCompletionChunk(
                id=chat_id,
                created=created,
                model=request.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatMessage(role=Role.ASSISTANT),
                        finish_reason="stop",
                        logprobs=None,
                    )
                ],
                **final_kwargs,
            )

        except Exception as e:
            logger.error(f"Error during stream generation: {str(e)}", exc_info=True)
            raise
