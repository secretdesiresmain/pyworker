import os
import logging
import dataclasses
import aiohttp
from typing import Optional, Union, Type

from aiohttp import web, ClientResponse

from lib.backend import Backend, LogAction
from lib.data_types import EndpointHandler
from lib.server import start_server
from .data_types import ComfyWorkflowData


MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://127.0.0.1:18288")

# This is the last log line that gets emitted once comfyui+extensions have been fully loaded
MODEL_SERVER_START_LOG_MSG = "To see the GUI go to: "
MODEL_SERVER_ERROR_LOG_MSGS = [
    "MetadataIncompleteBuffer",  # This error is emitted when the downloaded model is corrupted
    "Value not in list: ",  # This error is emitted when the model file is not there at all
    "[ERROR] Provisioning Script failed", # Error inserted by provisioning script if models/nodes fail to download
]


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

async def generate_client_response(
        client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        # Check if the response is actually streaming based on response headers/content-type
        is_streaming_response = (
            model_response.content_type == "text/event-stream"
            or model_response.content_type == "application/x-ndjson"
            or model_response.headers.get("Transfer-Encoding") == "chunked"
            or "stream" in model_response.content_type.lower()
        )

        if is_streaming_response:
            log.debug("Detected streaming response...")
            res = web.StreamResponse()
            res.content_type = model_response.content_type
            await res.prepare(client_request)
            async for chunk in model_response.content:
                await res.write(chunk)
            await res.write_eof()
            log.debug("Done streaming response")
            return res
        else:
            log.debug("Detected non-streaming response...")
            content = await model_response.read()
            return web.Response(
                body=content,
                status=model_response.status,
                content_type=model_response.content_type
            )
            

async def generate_async_client_response(
    client_request: web.Request, model_response: ClientResponse
) -> Union[web.Response, web.StreamResponse]:
    try: 
        result = await model_response.json()
        log.info(f"Result: {result}")
        log.info(f"Async job queued with id: {result.get('id', 'unknown')}")
        return web.json_response(
            result,
            status=model_response.status
        )
    except Exception as e:
        log.error(f"Error generating async client response: {e}")
        return web.json_response(
            {"error": str(e)},
            status=500
        )

@dataclasses.dataclass
class ComfyWorkflowHandler(EndpointHandler[ComfyWorkflowData]):

    @property
    def endpoint(self) -> str:
        return "/generate/sync"

    @property
    def healthcheck_endpoint(self) -> Optional[str]:
        return f"{MODEL_SERVER_URL}/health"

    @classmethod
    def payload_cls(cls) -> Type[ComfyWorkflowData]:
        return ComfyWorkflowData

    def make_benchmark_payload(self) -> ComfyWorkflowData:
        return ComfyWorkflowData.for_test()

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        return await generate_client_response(client_request, model_response)


@dataclasses.dataclass
class ComfyWorkflowAsyncHandler(EndpointHandler[ComfyWorkflowData]):
    """Handler for async image generation - returns run_id immediately"""

    @property
    def endpoint(self) -> str:
        return "/generate"  # ai-dock's async endpoint

    @property
    def healthcheck_endpoint(self) -> Optional[str]:
        return f"{MODEL_SERVER_URL}/health"

    @classmethod
    def payload_cls(cls) -> Type[ComfyWorkflowData]:
        return ComfyWorkflowData

    def make_benchmark_payload(self) -> ComfyWorkflowData:
        return ComfyWorkflowData.for_test()

    async def generate_client_response(
        self, client_request: web.Request, model_response: ClientResponse
    ) -> Union[web.Response, web.StreamResponse]:
        return await generate_async_client_response(client_request, model_response)


backend = Backend(
    model_server_url=MODEL_SERVER_URL,
    model_log_file=os.environ["MODEL_LOG"],
    allow_parallel_requests=False,
    benchmark_handler=ComfyWorkflowHandler(
        benchmark_runs=3, benchmark_words=100
    ),
    log_actions=[
        (LogAction.ModelLoaded, MODEL_SERVER_START_LOG_MSG),
        (LogAction.Info, "Downloading:"),
        *[
            (LogAction.ModelError, error_msg)
            for error_msg in MODEL_SERVER_ERROR_LOG_MSGS
        ],
    ],
)


async def handle_ping(_):
    return web.Response(body="pong")


async def handle_async_generate(request: web.Request):
    """
    Handle async generation requests by forwarding to ai-dock's /generate endpoint.
    The ai-dock ComfyUI API Wrapper handles async processing and webhooks.
    """
    try:
        # Parse request body
        data = await request.json()
        
        # Forward directly to ai-dock's /generate endpoint (async)
        # ai-dock will return immediately with request_id and handle webhooks
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{MODEL_SERVER_URL}/generate",
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                # Return ai-dock's response (contains request_id/run_id)
                result = await response.json()
                log.info(f"Result: {result}")
                log.info(f"Async job queued with id: {result.get('id', 'unknown')}")
                log.info(f"200 response")
                return web.json_response(
                    result,
                    status=200
                )
        
    except Exception as e:
        log.error(f"Error handling async request: {e}")
        return web.json_response(
            {"error": str(e)},
            status=500
        )


async def handle_job_status(request: web.Request):
    """
    Get status of an async job by forwarding to ai-dock's result endpoint.
    """
    request_id = request.match_info.get("run_id")
    
    try:
        # Forward to ai-dock's result endpoint
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{MODEL_SERVER_URL}/result/{request_id}",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                result = await response.json()
                return web.json_response(
                    result,
                    status=response.status
                )
    except Exception as e:
        log.error(f"Error getting job status: {e}")
        return web.json_response(
            {"error": str(e)},
            status=500
        )


async def handle_stream_generate(request: web.Request):
    """
    Handle streaming generation requests by forwarding to ai-dock's /generate/stream endpoint.
    Returns Server-Sent Events (SSE) with real-time progress updates.
    """
    try:
        data = await request.json()
        
        # Forward to ai-dock's streaming endpoint
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{MODEL_SERVER_URL}/generate/stream",
                json=data,
                timeout=aiohttp.ClientTimeout(total=None)  # No timeout for streaming
            ) as response:
                # Stream the response back to client
                stream_response = web.StreamResponse()
                stream_response.content_type = 'text/event-stream'
                stream_response.headers['Cache-Control'] = 'no-cache'
                stream_response.headers['X-Accel-Buffering'] = 'no'
                
                await stream_response.prepare(request)
                
                async for chunk in response.content.iter_any():
                    await stream_response.write(chunk)
                
                await stream_response.write_eof()
                return stream_response
                
    except Exception as e:
        log.error(f"Error handling stream request: {e}")
        return web.json_response(
            {"error": str(e)},
            status=500
        )


routes = [
    web.post("/generate/sync", backend.create_handler(ComfyWorkflowHandler())),
    web.post("/generate/async", backend.create_handler(ComfyWorkflowAsyncHandler())),
    web.post("/generate/stream", handle_stream_generate),
    web.get("/result/{run_id}", handle_job_status),
    web.get("/ping", handle_ping),
]

if __name__ == "__main__":
    start_server(backend, routes)
