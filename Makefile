.PHONY: setup run clean

setup:
	sudo apt install python3.12-venv -y
	python3 -m venv rayenv
	. rayenv/bin/activate && pip install --upgrade pip
	. rayenv/bin/activate && pip install pydantic ray[vllm] vllm

run:
	. rayenv/bin/activate && python main.py

clean:
	rm -rf rayenv
