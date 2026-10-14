# Security policy

local-bench runs code written by language models and processes datasets downloaded from the internet,
so we take sandbox and data-handling bugs seriously.

## Supported versions

| Version | Supported |
|---|---|
| `main` | ✅ |
| latest `0.x` release | ✅ |
| older releases | ❌ (please upgrade) |

## Reporting a vulnerability

**Don't open a public issue.** Report it privately through GitHub's
[private vulnerability reporting](https://github.com/shaansundar/local-bench/security/advisories/new)
(the **Security** tab, then **Report a vulnerability**).

Please include:

- what an attacker controls (model output, a dataset file, a benchmark adapter, a web page fetched by
  the search tool, …) and what they gain
- the steps to reproduce, with your macOS version and the commit you tested
- any proof of concept, kept minimal and harmless

This is a volunteer-maintained project. We aim to acknowledge reports within 7 days and to agree on a
fix and disclosure timeline with you. Reporters are credited in the advisory unless they prefer not to be.

## What's in scope

- **Sandbox escapes.** Model-written code or an opencode agent writing outside its workspace, reading the
  protected secret stores (`~/.ssh`, `~/.aws`, `~/.gnupg`, the `gh` config, `~/.netrc`, …), or reaching
  the network when its task has `network: none` or localhost only.
- **Untrusted data executed outside the sandbox.** Unsafe deserialization, archive path traversal
  ("zip slip"), or a dataset file that gets run during `fetch` or `load`.
- **Dashboard and API.** Path traversal through the run-file endpoints, exposure beyond `127.0.0.1`, or
  script injection from stored model output.
- **Secret leakage.** API keys from `.env` or the opencode config ending up in logs, transcripts,
  results, or the dashboard.

## What's out of scope

- A model producing wrong, harmful or biased answers. That is what benchmarks measure.
- Weaknesses in third-party benchmarks, datasets or upstream harnesses. Please report those upstream;
  tell us too if the suite should work around them.
- macOS `sandbox-exec` itself being deprecated or bypassable by kernel-level exploits. It's a
  containment layer for benchmark code, not a VM, and the README says so.
- Attacks that need an attacker to already control your user account or your `models.yaml`/`.env`.
