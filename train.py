import torch
from torch.utils.data import DataLoader
from new_model import GRUNetwork
from synthesizer import Synthesizer, SynthesizerDataset
import soundfile as sf

def train():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    # Data
    synth = Synthesizer("synthesizer_config.yaml")
    dataset = SynthesizerDataset(synth, num_samples=800) # Used to be 10000
    dataloader = DataLoader(dataset, batch_size=80, shuffle=True)

    model.train()
    for epoch in range(10):  # change to desired number of epochs
        for batch in dataloader:
            mic = batch["mic"].to(device)[:, :322].unsqueeze(1)
            target = batch["target"].to(device)[:, :161]  # Match model output shape

            h01 = torch.zeros(1, mic.shape[0], 322).to(device)
            h02 = torch.zeros(1, mic.shape[0], 322).to(device)

            output = model(mic, h01, h02)
            loss = loss_fn(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch == 9:
            sample_rate = synth.nearend_datasets.sample_rate
            target_out = batch['target'][0].detach().cpu().numpy()
            nearend_out = batch['nearend'][0].detach().cpu().numpy()
            mic_out = batch['mic'][0].detach().cpu().numpy()
            sf.write(r"./result/target.wav", target_out,sample_rate)
            sf.write(r"./result/nearend.wav", nearend_out, sample_rate)
            sf.write(r"./result/mic.wav", mic_out, sample_rate)

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),  # optional
        }, f'checkpoints/model_epoch_{epoch+1}.pt')

if __name__ == "__main__":
    train()
