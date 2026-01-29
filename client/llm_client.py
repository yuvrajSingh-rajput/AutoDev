import asyncio
from typing import Any, AsyncGenerator
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APIError
from client.response import TextDelta, TokenUsage, StreamEvent, EventType

class LLMClient:
    def __init__(self) -> None:
        self._client : AsyncOpenAI | None = None
        self._max_retries: int = 3

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
            base_url='https://openrouter.ai/api/v1',
            api_key='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',
        )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def chat_completion(
        self, 
        messages: list[dict[str, Any]], 
        stream: bool = True
    )-> AsyncGenerator[StreamEvent, None]:
        client = self.get_client()

        kwargs = {
            "model": "tngtech/deepseek-r1t2-chimera:free",
            "messages": messages,
            "stream": stream
        }

        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    async for event in self._stream_response(client, kwargs):
                        yield event
                else:
                    event = await self._non_stream_response(client, kwargs)
                    yield event
                    return
            except RateLimitError as e:
                if attempt < self._max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=EventType.ERROR,
                        error=f'Rate limit exceeded: {e}'
                    )
                    return  
            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=EventType.ERROR,
                        error=f'Connection error: {e}'
                    )  
                    return
            except APIError as e:
                yield StreamEvent(
                    type=EventType.ERROR,
                    error=f'API error: {e}'
                )  
                return

    async def _stream_response(
            self, 
            client: AsyncOpenAI, 
            kwargs: dict[str, Any]
        ) -> AsyncGenerator[StreamEvent, None]:
            
            stream = await client.chat.completions.create(**kwargs) 
            
            usage: TokenUsage | None = None
            finish_reason: str | None = None 

            async for chunk in stream:
                if chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        cached_tokens=(
                            chunk.usage.prompt_tokens_details.cached_tokens
                            if chunk.usage.prompt_tokens_details else None
                        )
                    )

                if not chunk.choices:
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta

                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                if delta.content:
                    yield StreamEvent(
                        type=EventType.TEXT_DELTA,
                        text_delta=TextDelta(delta.content),
                    )
            
            yield StreamEvent(
                type=EventType.MESSAGE_COMPLETE,
                finish_reason=finish_reason,
                usage=usage
            )

    async def _non_stream_response(
        self, 
        client: AsyncOpenAI, 
        kwargs: dict[str, Any], 
    ) -> StreamEvent:
        response = await client.chat.completions.create(**kwargs) 
        choice = response.choices[0]
        message = choice.message

        text = None
        if message.content:
            text_delta = TextDelta(content=message.content)

        usage = None
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens
            )
        
        return StreamEvent(
            type=EventType.MESSAGE_COMPLETE,
            text_delta=text_delta,
            finish_reason=choice.finish_reason,
            usage=usage
        )
