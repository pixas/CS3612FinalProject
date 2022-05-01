import argparse
from typing import List

import numpy as np
import torch
from model.model_fashion_mnist import VAE
from torchvision.utils import save_image


def make_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--hidden_dims", default=[512, 64], type=List[int])
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--ckpt_dir", default=None, type=str)
    parser.add_argument("--encoded_dim", default=32, type=int)
    args = parser.parse_args()
    return args

def main():
    args = make_argparser()
    ckpt_dir = args.ckpt_dir
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = VAE(args.encoded_dim, args.hidden_dims, args.dropout).to(device)
    model_parameter = torch.load(ckpt_dir)
    model.load_state_dict(model_parameter['model'])
    model.eval()

    with torch.no_grad():
        alpha = np.arange(0.1, 1.0, 0.1)
        z1 = torch.randn((args.batch_size, args.encoded_dim)).to(device)
        z2 = torch.randn((args.batch_size, args.encoded_dim)).to(device)
        
        x_prime1 = model.decode(z1)
        x_prime2 = model.decode(z2)
        save_image(x_prime1, "./images/vae-image1.png")
        save_image(x_prime2, "./images/vae-image2.png")
        for a in alpha:
            final_image = x_prime1 * a + (1 - a) * x_prime2
            save_image(final_image, "./images/vae-{:.2f}-merge.png".format(a))
        # image1 = x_prime1[0, 0].detach().cpu().numpy()
        # image2 = x_prime2[0, 0].detach().cpu().numpy()
        
        # plt.imshow(image1, 'gray')
        
        # plt.show()
        # plt.imshow(image2, 'gray')
        # plt.show()


if __name__ == "__main__":
    main()
