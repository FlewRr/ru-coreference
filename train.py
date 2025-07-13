import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import coref_loss
from dataset import collate_fn, RuCoCoDataset
from model import SpanBert

if __name__ == "__main__":
    dataset = RuCoCoDataset(data_dir="RuCoCo")
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

    model = SpanBert()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    for epoch in range(3):
        for batch in dataloader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gold_antecedents_batch = batch['gold_antecedents'].to(device)

            batch_span_starts = []
            batch_span_ends = []
            for mentions in batch['mentions']:
                starts = [span[0] for span in mentions]
                ends = [span[1] for span in mentions]
                batch_span_starts.append(starts)
                batch_span_ends.append(ends)

            mention_scores_batch, antecedent_scores_batch = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                span_starts=batch_span_starts,
                span_ends=batch_span_ends,
            )

            losses = []
            batch_size = len(mention_scores_batch)
            for b in range(batch_size):
                mention_scores = mention_scores_batch[b].unsqueeze(0)
                antecedent_scores = antecedent_scores_batch[b].unsqueeze(0)
                gold_antecedents = gold_antecedents_batch[b].unsqueeze(0)

                loss = coref_loss(mention_scores, antecedent_scores, gold_antecedents)
                losses.append(loss)

            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")
