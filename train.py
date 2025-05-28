import torch
from torch.utils.data import DataLoader, random_split
from new_model import GRUNetwork
from synthesizer import Synthesizer, SynthesizerDataset
import soundfile as sf


def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            mic = batch["mic"].to(device)[:, :322].unsqueeze(1)
            target = batch["target"].to(device)[:, :161]

            h01 = torch.zeros(1, mic.shape[0], 322).to(device)
            h02 = torch.zeros(1, mic.shape[0], 322).to(device)

            output, hn1, hn2 = model(mic, h01, h02)
            loss = loss_fn(output, target)
            total_loss += loss.item() * mic.size(0)
            
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def final_evaluate(model, dataloader, device, loss_fn, synth):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            mic = batch["mic"].to(device)[:, :322].unsqueeze(1)
            target = batch["target"].to(device)[:, :161]

            h01 = torch.zeros(1, mic.shape[0], 322).to(device)
            h02 = torch.zeros(1, mic.shape[0], 322).to(device)

            output, hn1, hn2 = model(mic, h01, h02)
            loss = loss_fn(output, target)
            total_loss += loss.item() * mic.size(0)
            last_batch = batch

    

    sample_rate = synth.nearend_datasets.sample_rate
    target_out = last_batch['target'][0].detach().cpu().numpy()
    nearend_out = last_batch['nearend'][0].detach().cpu().numpy()
    mic_out = last_batch['mic'][0].detach().cpu().numpy()
    denoised = output * mic[:, 0, :161]  # or nearend
    denoised_out = denoised[0].detach().cpu().numpy()
    sf.write(r"./final_example/target.wav", target_out, sample_rate)
    sf.write(r"./final_example/nearend.wav", nearend_out, sample_rate)
    sf.write(r"./final_example/mic.wav", mic_out, sample_rate)
    sf.write(r"./final_example/denoised.wav", denoised_out, sample_rate)
    print("Output min/max:", output.min().item(), output.max().item())
    print("Sample rate:", sample_rate)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    # Dataset and splitting
    synth = Synthesizer("synthesizer_config.yaml")
    full_dataset = SynthesizerDataset(synth, num_samples=800)

    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    valid_len = int(0.1 * total_len)
    test_len = total_len - train_len - valid_len

    train_dataset, valid_dataset, test_dataset = random_split(
        full_dataset, [train_len, valid_len, test_len]
    )

    batch_size = 80
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")

    for epoch in range(10):
        model.train()
        for batch in train_loader:
            mic = batch["mic"].to(device)[:, :322].unsqueeze(1)
            target = batch["target"].to(device)[:, :161]

            h01 = torch.zeros(1, mic.shape[0], 322).to(device)
            h02 = torch.zeros(1, mic.shape[0], 322).to(device)

            output, hn1, hn2 = model(mic, h01, h02)
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        val_loss = evaluate(model, valid_loader, device, loss_fn)
        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pt")
            

    # Test after training
    test_loss = final_evaluate(model, test_loader, device, loss_fn,synth)
    print(f"\nFinal Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    train()
