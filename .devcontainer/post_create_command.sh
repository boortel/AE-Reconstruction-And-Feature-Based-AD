apt-get update
apt-get install -y tmux
apt-get install -y nvtop

# making sure correct python version is installed in the env
conda install -y python==3.11.8
python3 -m pip install --upgrade pip
pip3 install --no-cache-dir -r requirements.txt
pip cache purge --no-input
conda clean -a -y

# opt-out of dvc data collection
dvc config core.analytics false
dvc pull

pre-commit install


if ! [ -d aliked ]; then
    echo "\033[31m ALIKED nenalezen, spust git clone git@gitlab.com:irafm-ai/aliked.git mimo container \033[0m"
    echo "\033[31m ALIKED ma vlastni custom ops, ktere je treba buildnout viz README. Build se provadi pouze jednou \033[0m"
    # aliked na zarovnavani obrazku ma nejake vlastni pytorch ops, ktere je treba buildnout
fi

if ! [ -d dinov2 ]; then
    echo "\033[31m Dinov2 nenalezen, spust git clone git@gitlab.com:irafm-ai/mepac_lisovani_plastu_dinov2.git && mv mepac_lisovani_plastu_dinov2 dinov2 mimo container \033[0m"
fi
