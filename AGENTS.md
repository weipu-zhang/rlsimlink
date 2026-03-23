# AGENTS.md — rlsimlink Project Guide

## What is rlsimlink?

A lightweight middleware that connects RL environment processes to trainer processes via Unix Domain Sockets. It allows an RL environment (e.g., Atari, DMLab, MineRL) running in one Python environment (Conda/Docker/venv) to be used by a trainer in a different environment on the same host.

## Key Concepts

- **Transport**: Unix Domain Socket at `/dev/shm/rlsimlink/<socket_id>/socket`
- **Observations**: Passed via `/dev/shm/rlsimlink/<socket_id>/obs` (numpy `.npy` file) to avoid large JSON payloads
- **Protocol**: JSON messages with 4-byte big-endian length prefix
- **Auto-launch**: If no `socket_id` is given, `RLEnv` spawns the server automatically via Conda (`conda run`) or Docker depending on `env-registry.conf`

## Project Layout

```
rlsimlink/                      # project root
├── env-registry.conf           # Maps env types to backend (conda/docker) + resource names
├── pyproject.toml              # Package metadata; client dep is numpy only
├── examples/
│   ├── atari.py                # End-to-end Atari usage example
│   ├── dmlab.py                # DMLab usage example
│   └── vizdoom.py              # VizDoom usage example
├── scripts/
│   ├── atari.sh                # Convenience launch scripts
│   ├── dmlab.sh
│   ├── vizdoom.sh
│   └── visualize.sh
└── rlsimlink/                  # installable package
    ├── cli.py                  # Entry point: `rlsimlink serve / test / docker`
    ├── utils.py                # Singleton logger with color-coded levels + snapshots
    ├── src/
    │   ├── client.py           # RLEnv — Gymnasium-style client
    │   ├── server.py           # RLEnvServer — Unix socket server
    │   ├── env_runtime.py      # EnvServerLauncher — spawns server (Conda or Docker)
    │   ├── socket_paths.py     # Path helpers for /dev/shm layout
    │   ├── common/
    │   │   ├── socket_manager.py   # SocketManager — send/recv JSON + obs over socket
    │   │   └── action_space.py     # Discrete / Continuous / Mixed action spaces
    │   └── envs/
    │       ├── __init__.py         # create_env_manager() factory
    │       ├── atari/              # AtariEnvManager — frame-skip/noop-reset wrappers
    │       ├── vizdoom/            # VizDoomEnvManager
    │       ├── minerl/             # MineRLEnvManager — bridges old Gym → new Gymnasium API
    │       ├── dmlab/              # DMLabEnvManager
    │       ├── dmc/                # DeepMind Control Suite (stub)
    │       └── metaworld/          # MetaWorld (stub)
    ├── env_requirements/
    │   ├── atari.txt
    │   └── vizdoom.txt
    └── docker/
        ├── docker.py               # Docker management utilities
        ├── utils.py
        ├── docker-compose.yml      # network_mode: host (required for /dev/shm sharing)
        └── dockerfiles/
            ├── Dockerfile.dmlab    # CUDA 12.8 + Ubuntu 24.04 + DMLab
            └── Dockerfile.minerl   # MineRL Docker image
```

## Data Flow

```
Client: RLEnv.reset()
  → SocketManager.send_json({"operation": "reset", "env_type": "atari", ...})
  ← SocketManager.receive_json({"status": "ok", "info": {...}})
  → load observation from /dev/shm/rlsimlink/<id>/obs (numpy)

Client: RLEnv.step(action)
  → send_json({"operation": "step", "action": [3]})
  ← receive_json({"reward": 1.0, "terminated": false, ...})
  → load new observation from /dev/shm
```

## Supported Environments

| Type | Status | Backend |
|------|--------|---------|
| Atari | Implemented | Conda (`gymnasium[atari]`, frame-skip / noop-reset wrappers) |
| VizDoom | Implemented | Conda (`vizdoom`) |
| MineRL | Implemented | Conda or Docker (`Dockerfile.minerl`) |
| DMLab | Implemented | Docker (`Dockerfile.dmlab`, CUDA 12.8) |
| DMC (DeepMind Control) | Stub | — |
| MetaWorld | Stub | — |

## Adding a New Environment

1. Create `rlsimlink/src/envs/<name>/` with an env manager class implementing `create()`, `reset()`, `step()`, `close()`, `get_action_space()`
2. Register it in `rlsimlink/src/envs/__init__.py` → `create_env_manager()`
3. Add an entry in `env-registry.conf` (backend = `conda` or `docker`)

## Runtime Requirements

- Linux/Unix with `/dev/shm` (shared memory filesystem)
- Python >= 3.8
- Client deps: `numpy` only
- Server deps: `numpy` + environment-specific libraries (e.g., `gymnasium[atari]`, `vizdoom`, `minerl`)
- Docker: required for docker-backed environments (DMLab; optional for MineRL)
- Conda: required for conda-backed environments (Atari, VizDoom, MineRL)
