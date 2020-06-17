mkdir ood_data
cd ood_data
# Download DTD (Texture dataset)
gdown https://drive.google.com/uc?id=1gh5N42fuiHbgAALvAdyohAFGiCckauMa
# download Places
gdown https://drive.google.com/uc?id=1VsT2Swn54ggnG8Y4Gelw-72NmfKtOErQ

# Download TinyImageNet (cropped)
wget https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz
tar -xvzf Imagenet.tar.gz
rm Imagenet.tar.gz

# Download TinyImageNet (resized)
wget https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz
tar -xvzf Imagenet_resize.tar.gz
rm Imagenet_resize.tar.gz
