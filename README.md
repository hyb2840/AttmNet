# AttmNet
![1](https://github.com/user-attachments/assets/ac671668-c7b6-4be1-bda7-10abe6627ba8)
# Environment install
```python 
git clone https://github.com/hyb2840/AttmNet.git
cd AttmNet
```
# Install mamba
```python 
cd mamba
python setup.py install
```
# Install monai
```python 
pip install monai
```
# train
```python
CUDA_VISIBLE_DEVICES=1 python train.py --arch AttmNet --dataset BUS --input_w 128 --input_h 128 --name BUS_AttmNet  --data_dir .../inputs/
```
# test
```python
python test.py --name BUS_AttmNet --output_dir .../outputs/
```
