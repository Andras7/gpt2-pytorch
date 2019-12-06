import torch

from lm.gpt2.data.data_loader import TextProcessor
from lm.gpt2.model import GPT2Model


def generate(start_sent, max_tokens=300, temperature=0.7, top_k=32):
    start_sent = text_processor.encode(start_sent)

    with torch.no_grad():
        for i in range(max_tokens):
            input_ids = torch.LongTensor(start_sent).unsqueeze(0).to(device)
            output = model(input_ids)

            word_weights = output.squeeze()[-1].div(temperature).exp().cpu()
            tops = word_weights.topk(top_k)
            word_idx = torch.multinomial(tops[0], 1)[0].item()
            word_idx = tops[1][word_idx].item()

            start_sent.append(word_idx)

    decoded = text_processor.decode(start_sent)
    print(decoded)


if __name__ == '__main__':
    device = "cuda:0"
    saved_model_path = "models/model_partial"
    start_sentence = "opel"

    model = GPT2Model().to(device)
    text_processor = TextProcessor(model.n_tokens, prefix="data/subwords")
    model.load_state_dict(torch.load(saved_model_path, map_location=device))
    model.eval()

    generate(start_sentence)
