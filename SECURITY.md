# Security Policy

## Supported versions

Security fixes land on the latest released minor version. Upgrade to the
current release on [PyPI](https://pypi.org/project/ai-api-unified/) before
reporting an issue, in case it is already fixed.

## Reporting a vulnerability

Report vulnerabilities privately through GitHub Security Advisories:

<https://github.com/davecthomas/ai-api-unified/security/advisories/new>

Please do not open a public issue for a suspected vulnerability, and please do
not include working exploit code in the first report.

Useful details:

- the version of `ai-api-unified` and the Python version
- which provider extra is installed, and the provider involved
- what an attacker could reach, and what access the attack assumes
- minimal steps to reproduce

You can expect an acknowledgement within a few days, and an assessment of
whether the report is accepted along with an intended fix version.

## Scope

This library is an access layer over third-party AI provider SDKs. Report here
anything in this package: credential handling, the PII redaction middleware,
the observability path, and dependency constraints that force a vulnerable
transitive version.

A vulnerability in a provider's own SDK or service belongs to that vendor. If
this library pins you to a version you cannot patch, that part is ours, and is
worth reporting here.

## Credentials

This library reads provider credentials from the environment and does not log
their values. If you find a path where a key, token, or a caller's prompt
content reaches a log or an exception message, treat it as a vulnerability and
report it.
