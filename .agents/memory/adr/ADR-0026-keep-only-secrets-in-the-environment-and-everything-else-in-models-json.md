# ADR-0026 Keep only secrets in the environment and everything else in models.json

Status: accepted
Date: 2026-10-16
Owners: dave-thomas
Must read: true
Supersedes: 
Superseded by: 
ai-generated: True
ai-tool: claude
ai-surface: claude-code
ai-executor: local-agent

Purpose: Finish the move to one configuration file with a published schema, leaving the environment for credentials
Derived from: owner requirements audit, configuration section (2026-10-16)

## Context

- After 3.5.0 every model selection lived in models.json, but provider
  settings (base URLs, Vertex project and location, the
  OpenAI-compatible server, the Azure speech region, the Nova Reel output
  bucket, strict deprecations) were still environment keys, and middleware
  lived in a separate YAML named by AI_MIDDLEWARE_CONFIG_PATH.
- The audit asked for one configuration with a published schema (3.10.0
  shipped the schema).

## Decision

- models.json gains `providers`, `middleware`, `videos.bedrock_output_s3_uri`
  and `strict_deprecations`. Each replaced environment key keeps working
  through the existing settings mapping: it still wins over the file and
  logs one deprecation warning naming the field, the same contract as the
  3.5.0 selection keys.
- The `middleware` section is the YAML file's list, validated entry by
  entry with the same skip-and-warn rule. AI_MIDDLEWARE_CONFIG_PATH, when
  set, still wins.
- `AWS_REGION` is not deprecated. AWS sets it on Lambda and other hosts,
  so a warning would fire on every such deployment. It wins without a
  warning, and `providers.aws.region` serves when it is unset; its default
  moved from the settings class to the callers so the file can be read.
- Secrets stay in the environment: API keys, AWS credentials, and
  GOOGLE_APPLICATION_CREDENTIALS.

## Consequences

- One file, validated by one schema and one command, describes everything a
  service needs except its credentials.
- A per-environment difference is a different models.json
  (AI_MODELS_CONFIG), or a deprecated environment key during migration.

## Related code paths

- src/ai_api_unified/ai_models_config.py
- src/ai_api_unified/util/env_settings.py
- src/ai_api_unified/middleware/middleware_config.py
- src/ai_api_unified/config_schema.py
- scripts/migrate2json.py
- env_template
