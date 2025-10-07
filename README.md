# DEIMv2_CPP
[DEIMv2] Real Time Object Detection Meets DINOv3 C++ and ONNX version

DEIMv2(https://github.com/Intellindust-AI-Lab/DEIMv2) is an evolution of the DEIM framework while leveraging the rich features from DINOv3. Our method is designed with various model sizes, from an ultra-light version up to S, M, L, and X, to be adaptable for a wide range of scenarios. Across these variants, DEIMv2 achieves state-of-the-art performance, with the S-sized model notably surpassing 50 AP on the challenging COCO benchmark.

We completed it from originally python onnx to windows onnx.
<img width="689" height="490" alt="image" src="https://github.com/user-attachments/assets/ef663580-682f-4fbc-b8fb-aae49a092c4e" />


## Steps
1. according to the guide of Setup of DEIMv2:
```C++
conda create -n deimv2 python=3.11 -y
conda activate deimv2
pip install -r requirements.txt
```

2. Then deploy the model
```C++
pip install onnx onnxsim
python tools/deployment/export_onnx.py --check -c configs/deimv2/deimv2_dinov3_${model}_coco.yml -r model.pth
```

3. The model can be download at the DEIMv2 website and dinov3 website as showed in the corresponding filepath.we use the middle-coco-model for showing.
<img width="1408" height="1038" alt="image" src="https://github.com/user-attachments/assets/945735a3-04e4-4372-844b-dd830e2b858b" />
<img width="1414" height="1030" alt="image" src="https://github.com/user-attachments/assets/ff7ae54b-f6e8-41af-be92-6b0987f7444d" />


4. Export onnx
pip install opencv-python
pip install onnxruntime(or GPU version), 
python tools/deployment/export_onnx.py --check -c configs/deimv2/deimv2_dinov3_m_coco.yml -r deimv2_dinov3_m_coco.pth
<img width="1422" height="1024" alt="image" src="https://github.com/user-attachments/assets/cde21f3c-c089-47e7-a1dc-2c63e3ede702" />



5. python onnx evacuation
We can demo in python onnx version:python tools/inference/onnx_inf.py --onnx deimv2_dinov3_m_coco.onnx --input image.jpg
<img width="2508" height="1484" alt="image" src="https://github.com/user-attachments/assets/0b320a93-621b-4cd5-89df-d7857ec5fa8d" />


6. CPP onnx evacuation
Use the reference coding above, we use VS2019 and onnx-1.181.
The CPU version running at the intel Ultra9-185H
![onnx_result](https://github.com/user-attachments/assets/732cdb08-f972-42ee-8917-8503af173278)
<img width="3120" height="1996" alt="image" src="https://github.com/user-attachments/assets/a0e0f265-92bb-4dde-a2e7-38cb21edaf01" />


7. CPP GPU-accererate onnx evacuation
Enable the two lines for ONNX cuda running.

<img width="3120" height="1996" alt="image" src="https://github.com/user-attachments/assets/55356a1b-ffc3-44e5-88da-b51492ff69c4" />

The GPU version running at GTX1060 and intel-I9-13900KF.
<img width="3100" height="1642" alt="image" src="https://github.com/user-attachments/assets/4115102d-f655-4c6d-a3ea-390ed27302df" />

## PS.
1. Due to the file size limited in Github, the onnx can download in this google share:https://drive.google.com/file/d/1nPKDHrotusQ748O1cQXJfi5wdShq6bKp/view?usp=drive_link
2. Step 4 cost much system memory, you can config more virtual memory in windows system to cover it
<img width="1347" height="1137" alt="image" src="https://github.com/user-attachments/assets/a4ca09b1-1109-4658-9cc0-e45ea4e4c8be" />
 

## Thanks
Our work is built upon DEIMv2 and DINOv3. Thanks for their great work!
  
