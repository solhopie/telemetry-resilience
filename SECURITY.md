# Security Policy

## Reporting a security issue

If you discover a security vulnerability in Telemetry Resilience, please do
not open a public issue. Instead, contact the project maintainer privately:

**[SECURITY-CONTACT — to be configured by the project owner]**

Include, where possible:

- a description of the issue and its potential impact,
- steps to reproduce (a minimal scenario YAML or command line),
- the Telemetry Resilience version and your platform / Python version.

You can expect an acknowledgement within a reasonable time. Until the issue
is fixed and a release is available, please do not disclose it publicly.

## Security model

Telemetry Resilience is a local-first CLI. Its security properties are:

- **Telemetry stays local.** Input files are read and corrupted copies are
  written on the user's machine. The tool requires no account and no API key,
  contacts no AI service, and uploads no analytics or usage telemetry
  anywhere.
- **Target execution is explicit.** The tool executes the user's target
  program only when explicitly requested via a CLI command (`test`, `suite`,
  or `campaign` with `-- <command>` arguments). Nothing runs a target
  program silently in the background.
- **YAML is data, not commands.** Scenario, suite, and campaign YAML files
  hold data and expectations only. The target command is taken exclusively
  from the CLI's `-- <command>` arguments and is never read from a YAML file,
  so a downloaded YAML file cannot smuggle a command for the tool to run.
- **No shell execution.** Target processes are launched with subprocess
  argument arrays, never `shell=True`, so command arguments are passed
  literally and cannot be reinterpreted by a shell.
- **Artifact directory ownership.** Run artifacts are written to a directory
  chosen by the user. The tool refuses to overwrite a non-empty directory
  unless it carries the telemetry-resilience ownership marker, so an
  unrelated directory is never destroyed. Case names are validated so
  artifacts cannot escape the artifacts directory.
- **Original input protections.** The tool never modifies or deletes the
  original telemetry input. `inject` writes new files alongside the input;
  `test`/`suite`/`campaign` inject into temporary copies.

## Scope notes

This tool helps you test how your software responds to bad telemetry. It is
not a safety certification: it does not certify that a physical system is
safe or reliable. Corrupted outputs are synthetic test data — never feed them
back as real telemetry. When reporting a security issue, remember that a
target program being executed under test runs with your own privileges; do
not test targets you do not trust.
