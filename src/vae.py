from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import scipy.io
from torch.nn import init
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='vae')
parser.add_argument('--batchsize', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 50)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_interval', type=int, default=200, 
                    help='how many batches to wait before save model')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)


x_dim = 70720
z_dim = 512
pca_dim = 2048
dim_heatmap = 64
n_joints = 17


z_path = '../dataset/Human3.6m/latent'
x_path = '../dataset/Human3.6m/heatmaps'

#pre-trained model
pre_model = '../results/vae/0_net_VAE.pth'

#model path
path = '../results/vae'

x_values = np.linspace(-1, 1, dim_heatmap)
fig_groundtruth_xy = plt.figure(num='ground truth xy')
fig_groundtruth_z = plt.figure(num='ground truth z')
fig_predicted_xy = plt.figure(num='predicted xy')
fig_predicted_z = plt.figure(num='predicted z')
fgxy = []
fgz = []
fpxy = []
fpz = []
#predicted xy
for i in range(1, n_joints):
    f = fig_predicted_xy.add_subplot(4,4,i)
    fpxy.append(f)
#ground truth xy
for i in range(1, n_joints):
    f = fig_groundtruth_xy.add_subplot(4,4,i)
    fgxy.append(f)
#predicted z
for i in range(1, n_joints):
    f = fig_predicted_z.add_subplot(4,4,i)
    fpz.append(f)
#ground truth z
for i in range(1, n_joints):
    f = fig_groundtruth_z.add_subplot(4,4,i)
    fgz.append(f)

plt.ion()


def updateplot(data):
    pxy, pz, gxy, gz = data     

    for i in range(n_joints - 1):
        fpxy[i].clear()
        fgxy[i].clear()
        fpz[i].clear()
        fgz[i].clear()

    for i in range(n_joints - 1):
        fpxy[i].imshow(pxy[i])
        fpz[i].plot(x_values, pz[i])
        fgxy[i].imshow(gxy[i])
        fgz[i].plot(x_values, gz[i])

    plt.pause(0.001)


def plot_heatmap_xy(outputs, inputs):
    size = dim_heatmap * dim_heatmap + dim_heatmap
    size_xy = dim_heatmap * dim_heatmap

    # we plot 16 joints here
    # k = 16
    outdata = outputs[size:].cpu().detach().numpy()
    indata = inputs[size:].cpu().detach().numpy()

    if len(indata) != (n_joints - 1) * (dim_heatmap * dim_heatmap + dim_heatmap):
        raise('dimension doesn\'t match!')

    pre_xy = np.zeros((n_joints - 1, dim_heatmap, dim_heatmap))
    pre_z = np.zeros((n_joints - 1, dim_heatmap))
    gro_xy = np.zeros((n_joints - 1, dim_heatmap, dim_heatmap))
    gro_z = np.zeros((n_joints - 1, dim_heatmap))
    for i in range(n_joints-1):
        pre_data = outdata[i*size:(i+1)*size]
        pre_xy[i] = np.resize(pre_data[:size_xy], (dim_heatmap, dim_heatmap))
        pre_z[i] = pre_data[size_xy:]
        gro_data = indata[i*size:(i+1)*size]
        gro_xy[i] = np.resize(gro_data[:size_xy], (dim_heatmap, dim_heatmap))
        gro_z[i] = gro_data[size_xy:]


    updateplot([pre_xy, pre_z, gro_xy, gro_z])


class poseDataset(Dataset):
    def __init__(self, z_path, x_path):
        self.paths_z = []
        self.paths_x = []
        subs = os.listdir(z_path)
        for s in subs:
            acts = os.listdir(os.path.join(z_path, s))
            for a in acts:
                basepath_z = os.path.join(z_path, s, a)
                basepath_x = os.path.join(x_path, s, a)
                filenames = os.listdir(basepath_z)
                filenames = [f[:-4] for f in filenames]
                filenames = [int(f) for f in filenames]
                filenames.sort()
                filenames = [str(f) for f in filenames]
                filenames = [f+'.mat' for f in filenames]
                for f in filenames:
                    path_z = os.path.join(basepath_z, f)
                    path_x = os.path.join(basepath_x, f)
                    self.paths_z.append(path_z)
                    self.paths_x.append(path_x)

        
    def __getitem__(self, index):
        current_path_z = self.paths_z[index]
        latent = scipy.io.loadmat(current_path_z)['latent'][0]
        current_path_x = self.paths_x[index]
        heatmap = scipy.io.loadmat(current_path_x)['heatmap'][0]

        return torch.tensor(latent), torch.tensor(heatmap)

    def __len__(self):
        return len(self.paths_z)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # encoder
        self.fc1 = nn.Linear(x_dim, pca_dim)
        self.fc2 = nn.Linear(pca_dim, pca_dim)
        self.fc3 = nn.Linear(pca_dim, pca_dim)
        self.fc41 = nn.Linear(pca_dim, z_dim)
        self.fc42 = nn.Linear(pca_dim, z_dim)
        # decoder
        self.fc5 = nn.Linear(z_dim, pca_dim)
        self.fc6 = nn.Linear(pca_dim, pca_dim)
        self.fc7 = nn.Linear(pca_dim, pca_dim)
        self.fc8 = nn.Linear(pca_dim, x_dim)

    def encode(self, x):
        #build encoder model
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        h3 = F.leaky_relu(self.fc3(h2))
        mu = self.fc41(h3)
        logvar = self.fc42(h3)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = torch.sigmoid(mu + eps * std)
        return z

    def decode(self, z):
        h4 = F.leaky_relu(self.fc5(z))
        h5 = F.leaky_relu(self.fc6(h4))
        h6 = F.leaky_relu(self.fc7(h5))
        return torch.sigmoid(self.fc8(h6))


    def forward(self, z):
        out = self.decode(z)
        return out


model = VAE().to(device)
model.load_state_dict(torch.load(pre_model))
optimizer = optim.Adam(model.parameters(), lr=args.lr)

pd = poseDataset(z_path, x_path)
train_loader = DataLoader(pd, batch_size=args.batchsize, shuffle=True)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x):
    BCELoss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCELoss

def train(epoch):
    model.train()
    model.fc1.weight.requires_grad=False
    model.fc1.bias.requires_grad=False
    model.fc2.weight.requires_grad=False
    model.fc2.bias.requires_grad=False
    model.fc3.weight.requires_grad=False
    model.fc3.bias.requires_grad=False
    model.fc41.weight.requires_grad=False
    model.fc41.bias.requires_grad=False
    model.fc42.weight.requires_grad=False
    model.fc42.bias.requires_grad=False
    train_loss = 0
    for batch_idx, (z, x) in enumerate(train_loader):
        z = z.to(device)
        x = x.to(device)
        optimizer.zero_grad()
        recon_x = model(z)
        loss = loss_function(recon_x, x)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(z),
                100. * batch_idx / len(train_loader),
                loss.item() / len(z)))
            plot_heatmap_xy(recon_x[0], x[0])

        if batch_idx % args.save_interval == 0:
            print('save latest model ...')
            torch.save(model.state_dict(), os.path.join(path, 'latest_net_VAE.pth'))

    # print('====> Epoch: {} Average loss: {:.4f}'.format(
    #       epoch, train_loss / len(train_loader.dataset)))


# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for i, (data, _) in enumerate(test_loader):
#             data = data.to(device)
#             recon_batch, mu, logvar = model(data)
#             test_loss += loss_function(recon_batch, data, mu, logvar).item()
#             if i == 0:
#                 n = min(data.size(0), 8)
#                 comparison = torch.cat([data[:n],
#                                       recon_batch.view(args.batch_size, 1, 28, 28)[:n]])

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        print('save model at the end of epoch ', epoch)
        torch.save(model.state_dict(), os.path.join(path, str(epoch) +'_net_VAE.pth'))
        #test(epoch)
        # with torch.no_grad():
        #     sample = torch.randn(64, 20).to(device)
        #     sample = model.decode(sample).cpu()




