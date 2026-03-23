# AGENTS.md — rlsimlink Project Guide

## What is rlsimlink?

A lightweight middleware that connects RL environment processes to trainer processes via Unix Domain Sockets. It allows an RL environment (e.g., Atari) running in one Python environment (Conda/Docker/venv) to be used by a trainer in a different environment on the same host.

## Key Concepts

- **Transport**: Unix Domain Socket at `/dev/shm/rlsimlink/<socket_id>/socket`
- **Observations**: Passed via `/dev/shm/<socket_id>/obs` (numpy `.npy` file) to avoid large JSON payloads
- **Protocol**: JSON messages with 4-byte big-endian length prefix
- **Auto-launch**: If no `socket_id` is given, `RLEnv` spawns the server automatically via `conda run`

## Project Layout

```
rlsimlink/
├── cli.py                  # Entry point: `rlsimlink serve`
├── env-registry.conf       # Maps env types to backend (conda/docker) + env name
├── rlsimlink/
│   ├── src/
│   │   ├── client.py           # RLEnv — Gymnasium-style client
│   │   ├── server.py           # RLEnvServer — Unix socket server
│   │   ├── env_runtime.py      # EnvServerLauncher — spawns server subprocess
│   │   ├── socket_paths.py     # Path helpers for /dev/shm layout
│   │   ├── common/
│   │   │   ├── socket_manager.py   # SocketManager — send/recv JSON over socket
│   │   │   └── action_space.py     # Discrete / Continuous / Mixed action spaces
│   │   └── envs/
│   │       ├── __init__.py         # create_env_manager() factory
│   │       ├── atari/              # AtariEnvManager (fully implemented)
│   │       ├── dmc/                # DeepMind Control Suite (stub)
│   │       ├── minecraft/          # MineRL (stub)
│   │       └── metaworld/          # MetaWorld (stub)
│   └── utils.py            # Singleton logger with color-coded levels
├── docker/
│   ├── Dockerfile.dmlab    # CUDA 12.8 + Ubuntu 24.04 + DMLab
│   └── docker-compose.yml  # network_mode: host (required for /dev/shm sharing)
└── examples/
    └── atari.py            # End-to-end usage example
```

## Data Flow

```
Client: RLEnv.reset()
  → SocketManager.send_json({"operation": "reset", "env_type": "atari", ...})
  ← SocketManager.receive_json({"status": "ok", "info": {...}})
  → load observation from /dev/shm/<id>/obs (numpy)

Client: RLEnv.step(action)
  → send_json({"operation": "step", "action": [3]})
  ← receive_json({"reward": 1.0, "terminated": false, ...})
  → load new observation from /dev/shm
```

## Supported Environments

| Type | Status | Backend |
|------|--------|---------|
| Atari | Implemented | Conda (`gymnasium`, frame-skip wrappers) |
| DMlab | Stub | Docker (`Dockerfile.dmlab`) |
| Minecraft | Stub | — |
| MetaWorld | Stub | — |

## Adding a New Environment

1. Create `rlsimlink/src/envs/<name>/` with an env manager class implementing `create()`, `reset()`, `step()`, `get_action_space()`
2. Register it in `rlsimlink/src/envs/__init__.py` → `create_env_manager()`
3. Add an entry in `env-registry.conf`

## Runtime Requirements

- Linux/Unix with `/dev/shm` (shared memory filesystem)
- Python >= 3.8
- Client deps: `numpy` only
- Server deps: `numpy` + environment-specific libraries (e.g., `gymnasium[atari]`)
