"""Lightbulb Computer prototype server.

Serves the projector web app from ./web and exposes one endpoint:

  POST /api/assist  {text, image?, pointer?, widgets?}
    -> {actions: [{name, input}]}

Claude sees the camera frame + the spoken/typed request and replies with
UI actions (tool calls) that the browser executes on the projected surface.

Dev mode: Claude is reached through the Claude Agent SDK, which drives the
locally installed Claude Code CLI and uses its login - no API key needed.
"""

import asyncio
import base64
import contextlib
import json
import os
import shutil
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKError,
    CLINotFoundError,
    ResultMessage,
    TextBlock,
    create_sdk_mcp_server,
    query,
    tool,
)

ROOT = Path(__file__).parent
WEB = ROOT / "web"
CAPTURES = ROOT / "captures"
PORT = int(os.environ.get("PORT", "8000"))
MODEL = os.environ.get("LB_MODEL")  # None = whatever your Claude Code is set to
EFFORT = os.environ.get("LB_EFFORT", "low")  # low = snappy voice replies
CLAUDE_CLI = os.environ.get("LB_CLAUDE_CLI") or shutil.which("claude")

# Use the Claude Code login on this machine, never an API key from the environment.
os.environ.pop("ANTHROPIC_API_KEY", None)

SYSTEM = """You are the brain of a "Lightbulb Computer": a projector + camera mounted above a table \
(or pointed at a wall). The user talks to you and points with their hands. You can see what the \
camera sees. Everything you show is projected onto the physical surface as small widgets.

Respond ONLY by calling the lightbulb tools. Make all the tool calls you need in parallel in a single \
turn, then reply with just "ok". Keep all projected text short - it is read from a distance. When the user says \
"this"/"that"/"here", they mean what they are pointing at; if a pointer is present it is drawn on \
the image as a magenta ring.

Image coordinates for `highlight` are normalized: x from 0 (left edge of the image) to 1 (right edge), \
y from 0 (top) to 1 (bottom). Be precise - the highlight is projected back onto the real object.

Latency-sensitive; begin your answer immediately."""


def obj(props, required=None):
    return {
        "type": "object",
        "properties": props,
        "required": required if required is not None else list(props),
        "additionalProperties": False,
    }


TOOL_SPECS = [
    (
        "answer_card",
        "Pin a short answer/explanation card on the surface. Use for questions, "
        "reading/translating/summarizing what is on the table, definitions, math help, etc.",
        obj({"title": {"type": "string"}, "body": {"type": "string", "description": "Max ~60 words. Plain text, newlines allowed."}}),
    ),
    (
        "highlight",
        "Project glowing rings onto real objects/places visible in the camera image "
        "(e.g. 'where is X on this map', 'which of these is ripe', 'point out the typo').",
        obj({
            "items": {
                "type": "array",
                "items": obj({
                    "label": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "radius": {"type": "number", "description": "Normalized radius, ~0.03 for a small thing."},
                }),
            }
        }),
    ),
    ("timer", "Start a countdown timer widget.", obj({"label": {"type": "string"}, "seconds": {"type": "integer"}})),
    (
        "recipe",
        "Show a step-by-step recipe/instructions widget the user can step through by pointing.",
        obj({
            "title": {"type": "string"},
            "ingredients": {"type": "array", "items": {"type": "string"}},
            "steps": {
                "type": "array",
                "items": obj({"text": {"type": "string"}, "timer_seconds": {"type": "integer", "description": "0 if none"}}),
            },
        }),
    ),
    (
        "map",
        "Show an interactive map the user can pan with pinch-drag.",
        obj({"place": {"type": "string", "description": "Geocodable place name."}, "zoom": {"type": "integer"}}),
    ),
    ("weather", "Show a weather widget for a place.", obj({"place": {"type": "string"}})),
    ("note", "Pin a sticky note / list on the surface.", obj({"text": {"type": "string"}})),
    ("clock", "Show a clock widget.", obj({})),
    (
        "clear",
        "Remove widgets. target='pointed' removes the widget under the user's finger, 'all' clears everything.",
        obj({"target": {"type": "string", "enum": ["pointed", "all"]}}),
    ),
    ("say", "Speak a very short spoken reply (under 15 words) in addition to or instead of widgets.", obj({"text": {"type": "string"}})),
]
ALLOWED_TOOLS = [f"mcp__lightbulb__{name}" for name, _, _ in TOOL_SPECS]


def make_tools(actions):
    """In-process MCP tools that just record the UI action for the browser to run."""
    def make(name, description, schema):
        @tool(name, description, schema)
        async def handler(args):
            actions.append({"name": name, "input": args})
            return {"content": [{"type": "text", "text": "shown"}]}
        return handler
    return [make(*spec) for spec in TOOL_SPECS]


async def assist(payload):
    content = []
    if payload.get("image"):
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": payload["image"]},
        })
    ctx = []
    if payload.get("pointer"):
        ctx.append("The user is pointing (magenta ring in the image).")
    if payload.get("widgets"):
        ctx.append("Widgets currently projected: " + ", ".join(payload["widgets"]))
    ctx.append(f'User said: "{payload.get("text", "")}"')
    content.append({"type": "text", "text": "\n".join(ctx)})

    async def prompt():
        yield {"type": "user", "message": {"role": "user", "content": content}, "parent_tool_use_id": None}

    actions = []
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM,
        model=MODEL,
        effort=EFFORT,
        tools=[],  # no built-in Claude Code tools (Bash, Read, ...) - only ours
        mcp_servers={"lightbulb": create_sdk_mcp_server("lightbulb", tools=make_tools(actions))},
        allowed_tools=ALLOWED_TOOLS,
        setting_sources=[],  # don't pick up CLAUDE.md / hooks / settings from disk
        max_turns=6,
        cli_path=CLAUDE_CLI,
        cwd=str(ROOT),
    )

    texts = []
    try:
        async with contextlib.aclosing(query(prompt=prompt(), options=options)) as stream:
            async for msg in stream:
                if isinstance(msg, AssistantMessage):
                    texts += [b.text.strip() for b in msg.content if isinstance(b, TextBlock) and b.text.strip()]
                elif isinstance(msg, ResultMessage) and msg.is_error and not actions:
                    raise ClaudeSDKError(f"Claude Code returned an error: {msg.result or msg.subtype}")
    except ClaudeSDKError:
        if not actions:
            raise
    if not actions and texts:
        actions.append({"name": "answer_card", "input": {"title": "", "body": " ".join(texts)}})
    return {"actions": actions}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        line = str(args[0] if args else "")
        if "/api/" in line and "/api/log" not in line:
            sys.stderr.write("%s\n" % (fmt % args))

    def do_POST(self):
        if self.path == "/api/debug":
            return self.save_capture()
        if self.path == "/api/log":
            entry = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            print(f"[browser {entry.get('t', '')}] {entry.get('msg', '')}", flush=True)
            self.send_response(204)
            self.end_headers()
            return
        if self.path != "/api/assist":
            self.send_error(404)
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            print(f"> {body.get('text')!r} image={'yes' if body.get('image') else 'no'}", flush=True)
            t0 = time.time()
            result = asyncio.run(assist(body))
            print(f"< {[a['name'] for a in result['actions']]} in {time.time() - t0:.1f}s", flush=True)
            status = 200
        except CLINotFoundError:
            result, status = {"error": "Claude Code CLI not found. Install it or set LB_CLAUDE_CLI."}, 500
        except ClaudeSDKError as e:
            result, status = {"error": f"Claude Code error: {e}. Is `claude` logged in?"}, 502
        data = json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


    def save_capture(self):
        """Debug frames from the S/R keys and calibration: raw + annotated JPEG and metadata."""
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        CAPTURES.mkdir(exist_ok=True)
        stem = time.strftime("%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}-{body.get('label', 'snap')}"
        (CAPTURES / f"{stem}.raw.jpg").write_bytes(base64.b64decode(body["raw"]))
        (CAPTURES / f"{stem}.annotated.jpg").write_bytes(base64.b64decode(body["annotated"]))
        (CAPTURES / f"{stem}.json").write_text(json.dumps(body.get("meta", {}), indent=1))
        print(f"captured {stem}", flush=True)
        data = json.dumps({"saved": stem}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"Lightbulb Computer on http://localhost:{PORT}  (claude={CLAUDE_CLI}, model={MODEL or 'Claude Code default'}, effort={EFFORT})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
