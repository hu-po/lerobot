# Tatbot

see [tatbot](https://github.com/hu-po/tatbot)

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[tatbot, intelrealsense]"
```

```bash
uv pip install pytest
uv run pytest \
    tests/cameras/test_ip_camera.py \
    tests/robots/test_tatbot.py \
    tests/teleoperators/test_atari_teleoperator.py
```