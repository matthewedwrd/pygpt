# PyGPT
## About
PyGPT is a small language model with the purpose to become an open-source counterweight against CloseAI's GPT models. It's written in Python, and uses PyTorch.
## Requirements
To get PyGPT up and running, you'll need a working installation of Python.
After you obtain a working instance of python, you'll want to obtain PyTorch. The easiest way to obtain PyTorch, is to install it with Python's package manager, `pip3`:
```sh
pip3 install torch torchvision torchaudio
```
## Running
To actually run PyGPT, you'll want to run PyGPT using this format:
```sh
python3 pygpt.py -d [cpu/cuda] -m [train/chat]
```
For both arguments, you must choose at least one option. Note that CUDA only works if you have an NVIDIA GPU.
You can use `train` option with `-m` flag to train PyGPT off of some basic text which you can place in a file named `input.txt`. After training, it'll produce a model file, called `model.pth`. You can chat with it using `chat` option with the `-m` flag.
## License
PyGPT is licensed under the MIT license.