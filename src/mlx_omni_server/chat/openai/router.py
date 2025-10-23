import json
from typing import Generator, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from mlx_omni_server.chat.mlx.chat_generator import ChatGenerator
from mlx_omni_server.chat.openai.openai_adapter import (
    OpenAIAdapter,
    _gather_vlm_captions_from_request,
)
from mlx_omni_server.chat.openai.schema import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionUsage,
    ChatMessage,
    Role,
)

router = APIRouter(tags=["chat—completions"])


# ---> OpenAI client > [create_chat_completion] > returns Chat/VLM response: /v1/chat/completions
@router.post("/chat/completions", response_model=ChatCompletionResponse)
@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion"""
    # First, short-circuit for vision-only responses so we don't construct a text model unnecessarily
    extra_params = request.get_extra_params()
    extra_body = extra_params.get("extra_body", {})
    vlm_model_name = (
        extra_body.get("vlm_model")
        or extra_body.get("vision_model")
        or request.model
    )

    if not request.stream:
        has_images, vlm_text = _gather_vlm_captions_from_request(request, vlm_model_name)
        if has_images and vlm_text:
            # Emit a direct VLM response without invoking the text generator
            message = ChatMessage(role=Role.ASSISTANT, content=vlm_text)
            completion = ChatCompletionResponse(
                id=f"chatcmpl-0",
                created=int(__import__("time").time()),
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
            return JSONResponse(content=completion.model_dump(exclude_none=True))

        # No short-circuit: build text model and proceed
        text_model = _create_text_model(
            request.model,
            extra_params.get("adapter_path"),
            extra_params.get("draft_model"),
        )
        completion = text_model.generate(request)
        return JSONResponse(content=completion.model_dump(exclude_none=True))

    async def event_generator() -> Generator[str, None, None]:
        # Short-circuit for VLM streaming: single delta + final stop (optionally with usage)
        has_images, vlm_text = _gather_vlm_captions_from_request(request, vlm_model_name)
        if has_images and vlm_text:
            created = int(__import__("time").time())
            chat_id = f"chatcmpl-0"
            first_chunk = ChatCompletionChunk(
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
            yield f"data: {json.dumps(first_chunk.model_dump(exclude_none=True))}\n\n"

            created = int(__import__("time").time())
            include_usage = bool(request.stream_options and request.stream_options.include_usage)
            final_kwargs = {}
            if include_usage:
                final_kwargs["usage"] = ChatCompletionUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    prompt_tokens_details=None,
                )

            final_chunk = ChatCompletionChunk(
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
            yield f"data: {json.dumps(final_chunk.model_dump(exclude_none=True))}\n\n"
            yield "data: [DONE]\n\n"
            return

        # No short-circuit: stream from text generator
        text_model = _create_text_model(
            request.model,
            extra_params.get("adapter_path"),
            extra_params.get("draft_model"),
        )
        for chunk in text_model.generate_stream(request):
            yield f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ---> create_chat_completion > [_create_text_model] > OpenAIAdapter for text generation
def _create_text_model(
    model_id: str,
    adapter_path: Optional[str] = None,
    draft_model: Optional[str] = None,
) -> OpenAIAdapter:
    """Create a text model based on the model parameters.

    Uses the shared wrapper cache to get or create ChatGenerator instance.
    This avoids expensive model reloading when the same model configuration
    is used across different requests or API endpoints.
    """
    # Get cached or create new ChatGenerator
    wrapper = ChatGenerator.get_or_create(
        model_id=model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model,
    )

    # Create OpenAIAdapter with the cached wrapper directly
    return OpenAIAdapter(wrapper=wrapper)


# Legacy caching variables removed - now using shared wrapper_cache
# This eliminates duplicate caching logic and enables sharing between endpoints
