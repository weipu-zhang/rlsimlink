# rlsimlink

`rlsimlink` is a lightweight middleware that links an RL environment process to a trainer via Unix sockets. The RL env can live in Docker, Conda, or a bare venv—as long as you can `pip install rlsimlink` inside that environment you can expose it to any other process on the same machine.

## Highlights

- **Environment agnostic** – works with any Python environment; no Docker daemon required.
- **Shared-memory transport** – `/dev/shm/rlsimlink/<socket-id>/` stores the Unix socket and numpy observations for fast handoff.
- **Deterministic socket IDs** – pass a short `--socket-id` when serving and use the same id on the client to connect.
- **Hash-code logging** – every socket id creates its own rotating logs under `rlsimlink/logs/`.

## Workflow

1. **On the RL environment machine (could be Docker/Conda/venv)**:
   ```bash
   pip install rlsimlink
   rlsimlink serve --socket-id abc123 --log-level INFO
   ```
   This opens `/dev/shm/rlsimlink/abc123/socket` and starts the RL server in the current Python interpreter (Gym, MineRL, etc. should already be installed here).

2. **On the training process** (can be another env/container on the same host):
   ```python
   from rlsimlink import RLEnv, ActionSpace

   env = RLEnv(
       env_type="atari",
       env_name="BoxingNoFrameskip-v4",
       # socket_id="abc123",  # optional: auto-generated if omitted
   )
   obs, info = env.reset()
   for _ in range(1000):
       action = env.action_space.sample()
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           obs, info = env.reset()
   env.close()
   ```

If you prefer to specify a full path instead of an id you can pass `--socket-path /dev/shm/custom/socket` to `serve` and `socket_path="/dev/shm/custom/socket"` to `RLEnv`. When you omit both `socket_id` and `socket_path`, `RLEnv` auto-generates a shared-memory directory, starts the Atari server in the `atari` conda environment via `conda run -n atari rlsimlink serve`, and cleans it up when you call `env.close()`.

## CLI

```bash
rlsimlink serve [--socket-id <id> | --socket-path <path>] [--log-level INFO]
```

- `--socket-id` – friendly identifier stored under `/dev/shm/rlsimlink/<id>/socket`.
- `--socket-path` – explicit Unix socket path (overrides `--socket-id`).
- `--log-level` – minimum log level (`ERROR`, `WARNING`, `SUCCESS`, `STEP`, `INFO`). Default `INFO`.

The `serve` command keeps running until you stop it with `Ctrl+C`.

## Notes

- `/dev/shm` must be shared between the serving environment and the trainer (e.g., same host or shared namespace).
- Only Unix sockets are supported for now (no TCP mode yet).
- Observations are serialized as `.npy` files in the same shared-memory directory to avoid large payloads on the socket.
