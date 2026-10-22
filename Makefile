PY ?= python
.PHONY: install test data train train-lite export
install:
	$(PY) -m pip install -e ".[all]"
test:
	$(PY) -m pytest
data:
	animatai prepare
train:
	animatai train --config configs/base.yaml --run runs/base
train-lite:
	animatai train --config configs/lite.yaml --run runs/lite
export:
	animatai export --model runs/lite/best.pt --out models/lite_onnx
