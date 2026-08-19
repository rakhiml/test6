# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.8.4-base

# This workflow only uses core ComfyUI nodes (the LTXV nodes live in
# comfy_api_nodes, which ships with ComfyUI), so no custom nodes are installed.

COPY api-workflow.json /api-workflow.json
COPY workflow.json /workflow.json

# The base image's /start.sh runs the GPU pre-flight check, puts ComfyUI-Manager
# in offline mode, launches ComfyUI and then execs /handler.py — so overriding
# /handler.py is all that is needed, and the inherited CMD ["/start.sh"] stands.
COPY handler.py /handler.py
