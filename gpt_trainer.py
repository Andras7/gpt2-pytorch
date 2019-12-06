import torch
from apex.fp16_utils import FP16_Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from lm.gpt2.data.data_loader import GPT2Dataset, collate, SortedBatchSampler
from lm.gpt2.lamb import Lamb
from lm.gpt2.model import GPT2Model

if __name__ == '__main__':
    lr = 7e-3
    wd = 1.2e-6
    batch_size = 8
    load_ver = 0
    device = "cuda:0"
    corpus_path = "TODO"

    model = GPT2Model()
    model = model.to(device)
    dataset = GPT2Dataset(corpus_path, model.n_positions + 1, model.n_tokens)
    sampler = SortedBatchSampler(dataset, batch_size, True, sort_key=lambda d: d.size(0))
    train_loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate, num_workers=1, pin_memory=True)

    optimizer = FP16_Optimizer(Lamb(model.parameters(), lr=lr, weight_decay=wd), dynamic_loss_scale=True)
    criterion = torch.nn.NLLLoss(ignore_index=0).to(device)

    print("\n\nTotal params: " + str(sum(p.numel() for p in model.parameters())))

    if load_ver != 0:
        model.load_state_dict(torch.load(f"models/model{load_ver}", map_location=device))
        optimizer.load_state_dict(torch.load(f"models/optimizer_state", map_location=device))

    running_loss = 0
    for iteration in range(1 + load_ver, 30):
        print("\n\n\nIteration: " + str(iteration))

        for i, contexts in enumerate(tqdm(train_loader, smoothing=0)):
            contexts = contexts.to(device)
            input_x = contexts[:, :-1]
            targets = contexts[:, 1:]
            optimizer.zero_grad()
            output = model(input_x)
            loss = criterion(output.view(output.shape[0] * output.shape[1], -1), targets.reshape(-1))
            optimizer.backward(loss)
            optimizer.clip_master_grads(5)
            optimizer.step()

            # Loss printer
            loss_value = loss.item()
            running_loss = loss_value if running_loss == 0 else (running_loss * 0.99 + loss_value * 0.01)
            if i % 300 == 0:
                print(f"Loss: {running_loss} Current loss: {loss_value}")

            # Partial saver
            if i > 0 and i % 30000 == 0:
                print("\n\nPartial save...")
                torch.save(model.state_dict(), f"models/model_partial")
                torch.save(optimizer.state_dict(), f"models/optimizer_state_partial")

        print("\n\nSave model...")
        torch.save(model.state_dict(), f"models/model" + str(iteration))
        torch.save(optimizer.state_dict(), f"models/optimizer_state")
