conda create -n videovla python=3.10 -y
conda activate videovla
pip install -r requirements.txt
pip install draccus timm 
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip3 install torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install "dlimp@git+https://github.com/moojink/dlimp_openvla"
pip install tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3
pip install numpy==1.26.4