"""Runs inside the Stirrup venv (sandboxed). Usage: stirrup_runner.py job.json

job: {base_url, api_key, model, max_tokens, context, prompt, input_files, output_dir, max_turns, exec_dir}
Writes <output_dir>/../result.json with the finish parameters and produced files.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
from pydantic import BaseModel, Field
from stirrup import Agent, Tool, ToolResult, ToolUseCountMetadata
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools.code_backends import LocalCodeExecToolProvider


class SearchParams(BaseModel):
    query: str = Field(description="Web search query")


class FetchParams(BaseModel):
    url: str = Field(description="URL to fetch")


def web_search(p: SearchParams) -> ToolResult[ToolUseCountMetadata]:
    from ddgs import DDGS

    for attempt in range(3):
        try:
            rows = DDGS().text(p.query, max_results=8)
            text = "\n\n".join(f"{r.get('title')}\n{r.get('href')}\n{r.get('body')}" for r in rows)
            return ToolResult(content=text or "(no results)", metadata=ToolUseCountMetadata())
        except Exception as e:
            err = str(e)
            time.sleep(3 * (attempt + 1))
    return ToolResult(content=f"search failed: {err}", metadata=ToolUseCountMetadata())


def fetch_url(p: FetchParams) -> ToolResult[ToolUseCountMetadata]:
    try:
        r = httpx.get(p.url, follow_redirects=True, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        import trafilatura

        text = trafilatura.extract(r.text, include_tables=True) or r.text[:20000]
        return ToolResult(content=text[:20000], metadata=ToolUseCountMetadata())
    except Exception as e:
        return ToolResult(content=f"fetch failed: {e}", metadata=ToolUseCountMetadata())


async def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    client = ChatCompletionsClient(model=job["model"], max_tokens=job["max_tokens"], context_window_tokens=job["context"],
                                   base_url=job["base_url"], api_key=job["api_key"], timeout=3600)
    tools = [LocalCodeExecToolProvider(temp_base_dir=job["exec_dir"]),
             Tool(name="web_search", description="Search the web (DuckDuckGo).", parameters=SearchParams, executor=web_search),
             Tool(name="fetch_url", description="Fetch a web page as text.", parameters=FetchParams, executor=fetch_url)]
    agent = Agent(client=client, name="gdpval", max_turns=job["max_turns"], tools=tools)
    out = Path(job["output_dir"])
    async with agent.session(output_dir=out, input_files=job["input_files"] or None) as session:
        finish, _history, metadata = await session.run(job["prompt"])
    files = [str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()]
    Path(job["result_path"]).write_text(json.dumps({"finish": str(finish)[:4000], "files": files,
                                                    "metadata": str(metadata)[:4000]}))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
