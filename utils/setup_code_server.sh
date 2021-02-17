# Zsh and oh my zsh
apt-get install -y zsh tree htop vim
echo "Yes" | sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# Setup micromamba
wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
mv bin/micromamba /bin
rm -rf bin/micromamba
/bin/micromamba shell init -s zsh -p ~/micromamba
source ~/.zshrc

micromamba activate
micromamba install -y mamba python=3.8 -c conda-forge
mamba install -y pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install matplotlib
mamba install -y ipykernel

# install requirements
pip install -r requirements.txt

# Sparse Linear
export CUDA=cu101
pip install torch-scatter==latest+${CUDA} torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-1.6.0.html

# Zsh
zsh
micromamba activate

