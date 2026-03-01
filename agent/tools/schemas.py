from __future__ import annotations

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_entity",
            "description": (
                "Look up entity memory by track ID or class name. "
                "Returns semantic descriptions and last-visible graph context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track_id": {
                        "type": "integer",
                        "description": "Tracked object ID in scene memory",
                    },
                    "class_name": {
                        "type": "string",
                        "description": "Object class label, for example 'person' or 'cup'",
                    },
                    "include_graph_context": {
                        "type": "boolean",
                        "description": "Include summary of the latest historical graph snapshot",
                        "default": True,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_scene",
            "description": (
                "Describe the current scene and optionally focus on a spatial region. "
                "Use this for broad scene lookup rather than a single known entity."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "enum": ["full", "left", "center", "right", "top", "bottom"],
                        "description": "Spatial region filter for visible entities",
                        "default": "full",
                    },
                    "max_entities": {
                        "type": "integer",
                        "description": "Maximum number of entities to return",
                        "default": 20,
                    },
                    "include_stale": {
                        "type": "boolean",
                        "description": (
                            "Include entities whose current text description is unavailable or stale"
                        ),
                        "default": False,
                    },
                },
                "required": [],
            },
        },
    },
]
