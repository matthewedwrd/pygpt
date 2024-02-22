import argparse
import random
import numpy
import torch

device = None

configuration = {
	"batch_size": 16,
	"block_size": 32,
	"max_iterations": 5000,
	"evaluation_interval": 100,
	"learning_rate": 1e-3,
	"evaluation_iterations": 200,
	"embedding_size": 64,
	"attention_head_count": 4,
	"layer_count": 4,
	"dropout": 0.0,
	"context_window_size": 3
}

class Head(torch.nn.Module):
	def __init__(self, head_size):
		super().__init__()
		self.key = torch.nn.Linear(configuration["embedding_size"], head_size, bias=False)
		self.query = torch.nn.Linear(configuration["embedding_size"], head_size, bias=False)
		self.value = torch.nn.Linear(configuration["embedding_size"], head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(configuration["block_size"], configuration["block_size"])))
		self.dropout = torch.nn.Dropout(configuration["dropout"])

	def forward(self, x):
		B, T, C = x.shape
		k = self.key(x)
		q = self.query(x)
		wei = q @ k.transpose(-2, -1) * C**-0.5
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
		wei = torch.nn.functional.softmax(wei, dim=-1)
		wei = self.dropout(wei)
		v = self.value(x)
		out = wei @ v
		return out

class MultiHeadAttention(torch.nn.Module):
	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = torch.nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = torch.nn.Linear(num_heads * head_size, configuration["embedding_size"])
		self.dropout = torch.nn.Dropout(configuration["dropout"])

	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.dropout(self.proj(out))
		return out

class FeedForward(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.net = torch.nn.Sequential(torch.nn.Linear(configuration["embedding_size"], 4 * configuration["embedding_size"]), torch.nn.ReLU(), torch.nn.Linear(4 * configuration["embedding_size"], configuration["embedding_size"]), torch.nn.Dropout(configuration["dropout"]))

	def forward(self, x):
		return self.net(x)

class Block(torch.nn.Module):
	def __init__(self):
		super().__init__()
		head_size = configuration["embedding_size"] // configuration["attention_head_count"]
		self.sa = MultiHeadAttention(configuration["attention_head_count"], head_size)
		self.ffwd = FeedForward()
		self.ln1 = torch.nn.LayerNorm(configuration["embedding_size"])
		self.ln2 = torch.nn.LayerNorm(configuration["embedding_size"])

	def forward(self, x):
		x = x + self.sa(self.ln1(x))
		x = x + self.ffwd(self.ln2(x))
		return x

class BigramLanguageModel(torch.nn.Module):
	def __init__(self, vocab_size):
		super().__init__()
		self.token_embedding_table = torch.nn.Embedding(vocab_size, configuration["embedding_size"])
		self.position_embedding_table = torch.nn.Embedding(configuration["block_size"], configuration["embedding_size"])
		self.blocks = torch.nn.Sequential(*[Block() for _ in range(configuration["layer_count"])])
		self.ln_f = torch.nn.LayerNorm(configuration["embedding_size"])
		self.lm_head = torch.nn.Linear(configuration["embedding_size"], vocab_size)

	def forward(self, index, targets=None):
		B, T = index.shape
		tok_emb = self.token_embedding_table(index)
		pos_emb = self.position_embedding_table(torch.arange(T, device=device))
		x = tok_emb + pos_emb
		x = self.blocks(x)
		x = self.ln_f(x)
		logits = self.lm_head(x)

		loss = None
		if targets is not None:
			logits_flat = logits.view(-1, logits.size(-1))
			targets_flat = targets.reshape(-1)
			loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
		return logits, loss

	def generate(self, index, max_new_tokens):
		for _ in range(max_new_tokens):
			index_cond = index[:, -configuration["block_size"]:]
			logits, _ = self(index_cond)
			logits = logits[:, -1, :]
			probs = torch.nn.functional.softmax(logits, dim=-1)
			index_next = torch.multinomial(probs, num_samples=1)
			index = torch.cat((index, index_next), dim=1)
		return index

def batch_data(data, batch_size, block_size):
	total_batch_size = batch_size * block_size
	n_batch = len(data) // total_batch_size
	data = data[:n_batch * total_batch_size]
	data = data.view(batch_size, -1)
	return data

def get_batch(source, i, block_size):
	seq_len = min(block_size, len(source) - 1 - i)
	data = source[:, i:i+seq_len]
	target = source[:, i+1:i+1+seq_len]
	return data, target

def main():
	argument_parser = argparse.ArgumentParser()
	argument_parser.add_argument("-d", "--driver", choices=["cuda", "cpu"], help="What should PyGPT use to run?")
	argument_parser.add_argument("-m", "--mode", choices=["train", "chat"], help="Should we train PyGPT, or should we just chat with it?")
	arguments = argument_parser.parse_args()

	device = arguments.driver

	if arguments.mode == "train":
		text = None
		with open("input.txt", "r", encoding="utf-8") as file:
			text = file.read()

		characters = sorted(list(set(text)))
		stoi = {ch: i for i, ch in enumerate(characters)}
		itos = {i: ch for i, ch in enumerate(characters)}
		data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
		train_length = int(0.9 * len(data))
		train_data, validation_data = data[:train_length], data[train_length:]

		model = BigramLanguageModel(len(stoi)).to(device)
		optimizer = torch.optim.AdamW(model.parameters(), lr=configuration["learning_rate"])

		train_data = batch_data(train_data, configuration["batch_size"], configuration["block_size"])
		validation_data = batch_data(validation_data, configuration["batch_size"], configuration["block_size"])

		for i in range(configuration["max_iterations"]):
			model.train()
			start_idx = (i * configuration["block_size"]) % (train_data.size(1) - configuration["block_size"])
			inputs, targets = get_batch(train_data, start_idx, configuration["block_size"])
			inputs, targets = inputs.to(device), targets.to(device)

			optimizer.zero_grad()
			_, loss = model(inputs, targets)
			loss.backward()
			optimizer.step()

			if i % configuration["evaluation_interval"] == 0:
				model.eval()
				with torch.no_grad():
					val_start_idx = numpy.random.randint(0, validation_data.size(1) - configuration["block_size"])
					val_inputs, val_targets = get_batch(validation_data, val_start_idx, configuration["block_size"])
					val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
					_, val_loss = model(val_inputs, val_targets)
					print(f"ITERATION: {i}, TRAINING LOSS: {loss.item():.4f}, VALIDATION LOSS: {val_loss.item():.4f}")

		torch.save(model.state_dict(), 'model.pth')
		print("TRAINING FINISHED!")
	elif arguments.mode == "chat":
		text = None
		with open("input.txt", "r", encoding="utf-8") as file:
			text = file.read()
		
		chars = sorted(list(set(text)))
		stoi = {ch: i for i, ch in enumerate(chars)}
		itos = {i: ch for i, ch in enumerate(chars)}

		model = BigramLanguageModel(len(stoi)).to(device)
		model.load_state_dict(torch.load('model.pth'))
		model.eval()

		conversation_history = []

		while True:
			input_text = input("YOU: ")
			if input_text.lower() == 'quit':
				break

			conversation_history.append(input_text)
			context = ' '.join(conversation_history[-configuration["context_window_size"]:])

			context_idx = torch.tensor([stoi.get(c, 0) for c in context], dtype=torch.long, device=device).unsqueeze(0)
			generated = model.generate(context_idx, max_new_tokens=50)
			response = ''.join([itos[i] for i in generated[0].tolist()])
			
			print(f"BOT: {response}")
			conversation_history = conversation_history[-2 * configuration["context_window_size"]:]

if __name__ == "__main__":
	main()