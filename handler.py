import base64
import copy
import binascii
import http.client
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import runpod


COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
COMFYUI_URL = os.getenv("COMFYUI_URL", f"http://{COMFYUI_HOST}:{COMFYUI_PORT}")
COMFYUI_DIR = os.getenv("COMFYUI_DIR", "/comfyui")
COMFYUI_PID_FILE = os.getenv("COMFYUI_PID_FILE", "/tmp/comfyui.pid")
WORKFLOW_PATH = os.getenv("WORKFLOW_PATH", "/api-workflow.json")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "1"))
COMFYUI_TIMEOUT = int(os.getenv("COMFYUI_TIMEOUT", "600"))
DEFAULT_IMAGE_NAME = os.getenv("DEFAULT_IMAGE_NAME", "runpod-input.png")

# LTXV runs on Comfy.org's API, not on the GPU. Those nodes read their
# credentials from the prompt's extra_data; worker-comfyui uses this env var
# name for the same purpose, so keep it identical.
COMFY_ORG_API_KEY = os.getenv("COMFY_ORG_API_KEY")

# Nodes that call out to the Comfy.org API and therefore need a key.
API_NODE_CLASS_TYPES = {
    "LtxvApiTextToVideo",
    "LtxvApiImageToVideo",
    "LtxApi25TextToVideo",
    "LtxApi25ImageToVideo",
    "LtxApi25AudioToVideo",
}

# ComfyUI drops connections while it is booting or busy; these are worth retrying.
TRANSIENT_ERRORS = (
    urllib.error.URLError,
    http.client.HTTPException,
    TimeoutError,
    ConnectionError,
)

_WHITESPACE = re.compile(r"\s+")

_comfyui_process = None
_client_id = str(uuid.uuid4())


class ComfyUIError(RuntimeError):
    """ComfyUI rejected a request or failed to execute a prompt."""


def _json_request(path, payload=None, timeout=30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        f"{COMFYUI_URL}{path}",
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
        if not body:
            return None
        return json.loads(body.decode("utf-8"))


def _http_error_body(exc):
    """ComfyUI puts the useful part (error, node_errors) in the error body."""
    try:
        body = exc.read().decode("utf-8", "replace").strip()
    except Exception:
        body = ""
    return body or str(exc)


def _multipart_request(path, fields, files, timeout=60):
    boundary = f"----runpod-comfyui-{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    for name, file_info in files.items():
        filename, content, content_type = file_info
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.extend(content)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        f"{COMFYUI_URL}{path}",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ComfyUIError(
            f"ComfyUI rejected the upload ({exc.code}): {_http_error_body(exc)}"
        ) from exc


def _comfyui_is_up():
    try:
        _json_request("/system_stats", timeout=5)
        return True
    except TRANSIENT_ERRORS:
        return False


def _start_comfyui():
    global _comfyui_process

    if _comfyui_process and _comfyui_process.poll() is None:
        return

    # The base image's /start.sh normally launches ComfyUI before handing over to
    # this handler. Only spawn our own instance when nothing is listening yet,
    # otherwise the second process dies on "address already in use".
    if os.path.exists(COMFYUI_PID_FILE) or _comfyui_is_up():
        return

    command = [
        "python3",
        "main.py",
        "--listen",
        "0.0.0.0",
        "--port",
        str(COMFYUI_PORT),
        "--disable-auto-launch",
    ]
    _comfyui_process = subprocess.Popen(command, cwd=COMFYUI_DIR)


def _ensure_comfyui_alive():
    """Fail fast instead of waiting out the full timeout on a dead ComfyUI."""
    if _comfyui_process is not None and _comfyui_process.poll() is not None:
        raise ComfyUIError(
            f"ComfyUI exited unexpectedly with code {_comfyui_process.returncode}"
        )


def _wait_for_comfyui():
    deadline = time.time() + COMFYUI_TIMEOUT
    last_error = None

    while time.time() < deadline:
        _ensure_comfyui_alive()
        try:
            _json_request("/system_stats", timeout=5)
            return
        except TRANSIENT_ERRORS as exc:
            last_error = exc
            time.sleep(POLL_INTERVAL)

    raise ComfyUIError(f"ComfyUI did not become ready within {COMFYUI_TIMEOUT}s: {last_error}")


def _load_workflow():
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as workflow_file:
        return json.load(workflow_file)


def _decode_base64_image(image_value):
    if not isinstance(image_value, str) or not image_value.strip():
        raise ValueError("input.image must be a non-empty base64 string")

    mime_type = "image/png"
    if image_value.startswith("data:"):
        header, _, image_value = image_value.partition(",")
        mime_type = header.split(";")[0][len("data:"):] or "image/png"

    # Real payloads arrive line-wrapped, url-safe encoded or unpadded; strict
    # decoding rejects all three, so normalise before validating.
    payload = _WHITESPACE.sub("", image_value).replace("-", "+").replace("_", "/")
    payload += "=" * (-len(payload) % 4)

    try:
        decoded = base64.b64decode(payload, validate=True)
    except binascii.Error as exc:
        raise ValueError("input.image is not valid base64") from exc

    if not decoded:
        raise ValueError("input.image decoded to an empty image")
    return decoded, mime_type


def _upload_image(image_value, filename):
    image_bytes, mime_type = _decode_base64_image(image_value)
    result = _multipart_request(
        "/upload/image",
        {"type": "input", "overwrite": "true"},
        {"image": (filename, image_bytes, mime_type)},
    )
    name = result.get("name", filename)
    subfolder = result.get("subfolder", "")
    # LoadImage addresses an upload inside a subfolder as "subfolder/name".
    return f"{subfolder}/{name}" if subfolder else name


def _find_nodes(workflow, class_names):
    return [
        node_id
        for node_id, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type") in class_names
    ]


def _set_first_matching_input(workflow, class_names, input_name, value):
    for node_id in _find_nodes(workflow, class_names):
        inputs = workflow[node_id].setdefault("inputs", {})
        if input_name in inputs:
            inputs[input_name] = value
            return True
    return False


def _apply_uploaded_image(workflow, uploaded_name):
    load_image_nodes = _find_nodes(workflow, {"LoadImage"})
    if not load_image_nodes:
        raise ValueError("input.image was provided, but the workflow has no LoadImage node")

    for node_id in load_image_nodes:
        workflow[node_id].setdefault("inputs", {})["image"] = uploaded_name


def _get_lora_inputs(job_input):
    if "loras" in job_input:
        loras = job_input["loras"]
    elif "lora" in job_input:
        loras = job_input["lora"]
    elif "lora_name" in job_input:
        loras = job_input["lora_name"]
    else:
        return []

    if isinstance(loras, (str, dict)):
        return [loras]
    return loras


def _apply_loras(workflow, loras):
    if not loras:
        return

    if not isinstance(loras, list):
        raise ValueError("input.loras must be a list of LoRA names or objects")

    lora_nodes = _find_nodes(workflow, {"LoraLoader", "LoRALoader", "LoraLoaderModelOnly"})
    if not lora_nodes:
        raise ValueError(
            "input.loras was provided, but the workflow has no LoRA loader nodes. "
            "Use input.node_inputs with a custom workflow that includes LoraLoader nodes."
        )

    for index, lora in enumerate(loras):
        if index >= len(lora_nodes):
            raise ValueError("More LoRAs were provided than the workflow has LoRA loader nodes")

        node_id = lora_nodes[index]
        inputs = workflow[node_id].setdefault("inputs", {})
        if isinstance(lora, str):
            inputs["lora_name"] = lora
        elif isinstance(lora, dict):
            inputs["lora_name"] = lora.get("name") or lora.get("lora_name")
            if not inputs["lora_name"]:
                raise ValueError("LoRA objects must include name or lora_name")
            if "strength" in lora:
                inputs["strength_model"] = lora["strength"]
                inputs["strength_clip"] = lora["strength"]
            if "strength_model" in lora:
                inputs["strength_model"] = lora["strength_model"]
            if "strength_clip" in lora:
                inputs["strength_clip"] = lora["strength_clip"]
        else:
            raise ValueError("Each LoRA must be a string or object")


def _apply_inputs(workflow, job_input):
    custom_workflow = job_input.get("workflow")
    workflow = copy.deepcopy(custom_workflow if custom_workflow else workflow)
    if not isinstance(workflow, dict):
        raise ValueError("input.workflow must be a ComfyUI API-format workflow object")

    image_value = job_input.get("image") or job_input.get("image_base64")
    if image_value:
        filename = job_input.get("image_filename", DEFAULT_IMAGE_NAME)
        _apply_uploaded_image(workflow, _upload_image(image_value, filename))
    elif not custom_workflow and _find_nodes(workflow, {"LoadImage"}):
        raise ValueError("input.image is required for the default image-to-video workflow")

    simple_inputs = {
        "prompt": job_input.get("prompt"),
        "seed": job_input.get("seed"),
        "model": job_input.get("model"),
        "duration": job_input.get("duration"),
        "model.duration": job_input.get("duration"),
        "resolution": job_input.get("resolution"),
        "model.resolution": job_input.get("resolution"),
        "fps": job_input.get("fps"),
        "model.fps": job_input.get("fps"),
        "generate_audio": job_input.get("generate_audio"),
        "model.generate_audio": job_input.get("generate_audio"),
    }
    target_nodes = {
        "LtxApi25TextToVideo",
        "LtxApi25ImageToVideo",
        "LtxvApiTextToVideo",
        "LtxvApiImageToVideo",
        "LTXVImgToVideo",
    }
    for input_name, value in simple_inputs.items():
        if value is not None:
            _set_first_matching_input(workflow, target_nodes, input_name, value)

    _apply_loras(workflow, _get_lora_inputs(job_input))

    for node_id, inputs in job_input.get("node_inputs", {}).items():
        node = workflow.get(str(node_id))
        if node is None:
            raise ValueError(
                f"input.node_inputs references node {node_id}, which is not in the workflow"
            )
        node.setdefault("inputs", {}).update(inputs)

    return workflow


def _resolve_api_key(job_input):
    api_key = job_input.get("comfy_org_api_key") or COMFY_ORG_API_KEY
    return api_key or None


def _require_api_key(workflow, api_key):
    if api_key:
        return

    api_nodes = _find_nodes(workflow, API_NODE_CLASS_TYPES)
    if api_nodes:
        class_types = sorted({workflow[node_id]["class_type"] for node_id in api_nodes})
        raise ValueError(
            f"This workflow uses Comfy.org API nodes ({', '.join(class_types)}), which "
            "require an API key. Set the COMFY_ORG_API_KEY environment variable on the "
            "endpoint, or pass input.comfy_org_api_key with the request."
        )


def _queue_prompt(workflow, api_key=None):
    payload = {
        "prompt": workflow,
        "client_id": _client_id,
    }
    if api_key:
        # ComfyUI reads API-node credentials from extra_data.
        payload["extra_data"] = {"api_key_comfy_org": api_key}

    try:
        result = _json_request("/prompt", payload)
    except urllib.error.HTTPError as exc:
        raise ComfyUIError(
            f"ComfyUI rejected the workflow ({exc.code}): {_http_error_body(exc)}"
        ) from exc

    if not result or "prompt_id" not in result:
        raise ComfyUIError(f"ComfyUI did not return a prompt_id: {result!r}")
    return result["prompt_id"]


def _check_history_status(entry):
    """A failed prompt still lands in history, with empty outputs."""
    status = entry.get("status") or {}
    if status.get("status_str") == "error" or status.get("completed") is False:
        messages = json.dumps(status.get("messages", []))[:4000]
        raise ComfyUIError(f"ComfyUI failed to execute the prompt: {messages}")
    return entry


def _wait_for_history(prompt_id):
    deadline = time.time() + COMFYUI_TIMEOUT
    while time.time() < deadline:
        _ensure_comfyui_alive()
        try:
            history = _json_request(f"/history/{prompt_id}", timeout=30)
        except TRANSIENT_ERRORS:
            history = None
        if history and prompt_id in history:
            return _check_history_status(history[prompt_id])
        time.sleep(POLL_INTERVAL)

    raise ComfyUIError(f"ComfyUI prompt {prompt_id} did not finish within {COMFYUI_TIMEOUT}s")


def _download_output(file_info):
    query = urllib.parse.urlencode(
        {
            "filename": file_info["filename"],
            "subfolder": file_info.get("subfolder", ""),
            "type": file_info.get("type", "output"),
        }
    )
    with urllib.request.urlopen(f"{COMFYUI_URL}/view?{query}", timeout=60) as response:
        content = response.read()
        mime_type = response.headers.get_content_type()

    return {
        "filename": file_info["filename"],
        "subfolder": file_info.get("subfolder", ""),
        "type": file_info.get("type", "output"),
        "mime_type": mime_type,
        "data": base64.b64encode(content).decode("ascii"),
    }


def _collect_outputs(history):
    outputs = []
    # SaveVideo reports its file under "images" (ui.PreviewVideo); the other keys
    # cover VHS/audio nodes in custom workflows.
    for node_output in history.get("outputs", {}).values():
        for key in ("images", "gifs", "videos", "audio"):
            for file_info in node_output.get(key, []):
                output = _download_output(file_info)
                output["kind"] = key
                outputs.append(output)
    return outputs


def handler(job):
    job_input = job.get("input", {})
    api_key = _resolve_api_key(job_input)

    _start_comfyui()
    _wait_for_comfyui()

    workflow = _apply_inputs(_load_workflow(), job_input)
    _require_api_key(workflow, api_key)

    prompt_id = _queue_prompt(workflow, api_key)
    history = _wait_for_history(prompt_id)
    outputs = _collect_outputs(history)

    if not outputs:
        raise ComfyUIError(
            f"ComfyUI prompt {prompt_id} completed but produced no output files. "
            "Check that the workflow contains a save node."
        )

    return {
        "prompt_id": prompt_id,
        "outputs": outputs,
    }


runpod.serverless.start({"handler": handler})
