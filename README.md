# Copilot-AI-Proxy

Small FastAPI proxy for using custom AI endpoints with Copilot Chat custom model support.

It exposes one local OpenAI-compatible base URL:

```text
http://127.0.0.1:8787/v1
```

## Routes

Routes are defined in [config.json](config.json). The incoming model ID must match a key under `routes`.

Supported route providers:

- `openai`: forwards OpenAI-compatible chat completion requests.
- `anthropic`: converts OpenAI chat completion requests to Anthropic Messages API requests, then converts responses back.

Config examples:

- [config.sample.json](config.sample.json): placeholder template.
- [config.azure-example.json](config.azure-example.json): near-real Azure Foundry example without secrets.

## Azure Foundry Naming

Use the deployment name from Foundry as the model name. This is often, but not always, the same as the base model name.

- OpenAI base URL: `https://<resource-name>.openai.azure.com/openai/v1`
- Anthropic base URL: `https://<resource-name>.services.ai.azure.com/anthropic/v1`
- `upstream_model`: the Foundry deployment name
- route key: the model ID VS Code sends; usually keep it the same as `upstream_model`

Do not use the Azure portal URL or Foundry project URL as `upstream_base_url`.

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## API Keys

Prefer environment variables instead of putting real keys in [config.json](config.json):

```json
"upstream_api_key_env": "AZURE_FOUNDRY_API_KEY"
```

PowerShell example:

```powershell
$env:AZURE_FOUNDRY_API_KEY="your-key"
$env:ANTHROPIC_API_KEY="your-key"
```

## Run

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8787
```

Linux/macOS:

```bash
.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8787
```

## VS Code

Add custom models in VS Code's user `chatLanguageModels.json` file.

Common locations:

- VS Code Insiders on Windows: `%APPDATA%\Code - Insiders\User\chatLanguageModels.json`
- VS Code Stable on Windows: `%APPDATA%\Code\User\chatLanguageModels.json`
- Linux: `~/.config/Code/User/chatLanguageModels.json`
- macOS: `~/Library/Application Support/Code/User/chatLanguageModels.json`

Each custom model should point to the local proxy:

```json
[
	{
		"name": "Azure Foundry",
		"vendor": "customendpoint",
		"apiKey": "123",
		"apiType": "chat-completions",
		"models": [
			{
				"id": "gpt-5.5",
				"name": "gpt-5.5",
				"url": "http://127.0.0.1:8787/v1",
				"toolCalling": true,
				"vision": true,
				"maxInputTokens": 128000,
				"maxOutputTokens": 16000
			},
			{
				"id": "claude-sonnet-4-6",
				"name": "claude-sonnet-4-6",
				"url": "http://127.0.0.1:8787/v1",
				"toolCalling": true,
				"vision": true,
				"maxInputTokens": 128000,
				"maxOutputTokens": 16000
			}
		]
	}
]
```

The `id` must match a route key in [config.json](config.json).

## Quick Check

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8787/v1/models" -Method Get | ConvertTo-Json -Depth 20
```
