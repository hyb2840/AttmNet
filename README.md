# AttmNet

![1](https://github.com/user-attachments/assets/e58fd538-7de4-4c6c-aca9-fcce0ed2fa03)
![Fig3（改）](https://github.com/user-attachments/assets/c9bf7ab5-a866-4477-ac86-52e460380b72)

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
