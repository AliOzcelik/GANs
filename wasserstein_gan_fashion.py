import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import torchvision
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
    
n_epochs = 100
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5
device = 'mps'
    
path = r"/Users/desidero/Desktop/Kodlar/GANs/wgan_fashion_2/"
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize(0.5, 0.5)])

data = FashionMNIST(root=path, train=False, download=True, transform=transform)
data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)


def get_noise(n_samples, z_dim, device='mps'):
    return torch.randn(n_samples, z_dim, device=device)


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)
    
    
class Critic(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
crit = Critic().to(device) 
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))


gen = gen.apply(weights_init)
crit = crit.apply(weights_init)

criterion = nn.BCELoss().to(device)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
                                
gen = gen.apply(weights_init)
crit = crit.apply(weights_init)


def get_gradient(crit, real, fake, epsilon):
    mixed_images = real * epsilon + fake * (1 - epsilon)
    mixed_scores = crit(mixed_images)
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores), 
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1)**2)
    return penalty



def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)
    return gen_loss  

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    return crit_loss


a = 1
crit_losses = []
gen_losses = []
for epoch in range(100):
    print('Epoch: ', epoch+1)
    
    for i, (real, _) in enumerate(data_loader, 0):
        cur_batch_size = len(real)
        real = real.to(device)
        
        mean_iteration_critic_loss = 0
        for _ in range(crit_repeats):
            crit_opt.zero_grad()
            noise = get_noise(cur_batch_size, z_dim)
            fake = gen(noise)
            crit_fake_pred = crit(fake)
            crit_real_pred = crit(real)
            
            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(crit, real, fake, epsilon)
            gp = gradient_penalty(gradient)
            
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)
        
            crit_loss.backward()
            crit_opt.step()
            mean_iteration_critic_loss += crit_loss.item() / crit_repeats
    
        gen_opt.zero_grad()
        noise_2 = get_noise(cur_batch_size, z_dim)
        fake_2 = gen(noise_2)
        crit_fake_pred_2 = crit(fake_2)
        gen_loss = get_gen_loss(crit_fake_pred_2)
        
        gen_loss.backward()
        gen_opt.step()
    
        crit_losses.append(mean_iteration_critic_loss)
        gen_losses.append(gen_loss)
        
        
        if i % 100 == 0:
            print("Loss_D: %.4f, Loss_G: %.4f" % (mean_iteration_critic_loss, gen_loss))
            torchvision.utils.save_image(real, '{}/real_samples{}.png'.format(path, a), normalize=True)
            noise = get_noise(cur_batch_size, z_dim)
            fake = gen.forward(noise)
            torchvision.utils.save_image(fake.data, '{}/fake_samples{}.png'.format(path, a), normalize=True)
            a += 1


