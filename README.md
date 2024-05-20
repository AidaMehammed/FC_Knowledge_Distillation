# [FeatureCloud Knowledge Distillation](https://featurecloud.ai/app/knowledge_distillation)
### Image Classification with Knowledge Distillation

This App  facilitates the compression of deep neural network models through knowledge distillation. This technique transfers knowledge from a larger, complex model (the "teacher") to a smaller, simpler model (the "student"), resulting in reduced memory and computational requirements while maintaining performance.


Image classification is a fundamental task in computer vision, and this app caters to datasets like CIFAR and MNIST. CIFAR-10 and MNIST are widely used benchmark datasets for image classification tasks. CIFAR-10 consists of 60,000 32x32 color images in 10 classes, while MNIST comprises 28x28 grayscale images of handwritten digits.

![states diagram](dl/state_diagram.png)



## Config Settings
### Training Settings
```python
teacher_model: teacher.py
student_model: student.py

train_dataset: "train_dataset.pth"
test_dataset: "test_dataset.pth"

batch_size: 256
learning_rate: 0.001

max_iter: 10
```
### Training Options
#### Models
File name will provided as generic data to clients, which later will be imported by the app. The model class should have the name 'Model' and include the forward method. For more details, please refer to the examples provided in [models/pytorch/models](/data/sample_data/generic/cnn.py) 

`teacher_model`: This field should specify the Python file containing the teacher model implementation. It is expected to be in the .py format.

`student_model`: Similarly, specify the Python file containing the student model implementation.


#### Local dataset

`train_dataset` :  Path to the training dataset.

`test_dataset:` :  Path to the training dataset.

These datasets will be loaded using the `torch.utils.data.DataLoader` class.


#### Training config
`batch_size`: Specifies the number of samples in each training batch.

`learning_rate`: Determines the rate at which the model's parameters are updated during training.

`max_iter` : Defines the maximum number of communication rounds.


### Knowledge Distillation Settings 
```python

temperature : 2.0
alpha: 0.5
```

#### Distillation Hyper-Parameters config
`temperature`: Controls the softness of the teacher's logits during distillation. Higher values make the teacher's distribution softer, allowing for smoother knowledge transfer to the student.

`alpha`: Controls the weight assigned to the distillation loss compared to the standard cross-entropy loss. A higher alpha places more emphasis on mimicking the teacher's output distribution, while a lower alpha prioritizes fitting the true labels.


### Run app

#### Prerequisite

To run the model compression app, you should install Docker and FeatureCloud pip package:

```shell
pip install featurecloud, torchvision,bios
```

Then either download the fc-knowledge-distillation app image from the FeatureCloud docker repository:

```shell
featurecloud app download featurecloud.ai/fc-knowledge-distillation
```

Or build the app locally:

```shell
featurecloud app build featurecloud.ai/fc-knowledge-distillation
```

Please provide example data so others can run the fc-knowledge-distillation app with the desired settings in the `config.yml` file.

#### Run the model compression app in the test-bed

You can run the fc-knowledge-distillation app as a standalone app in the [FeatureCloud test-bed](https://featurecloud.ai/development/test) or [FeatureCloud Workflow](https://featurecloud.ai/projects). You can also run the app using CLI:

```shell
featurecloud test start --app-image featurecloud.ai/fc-knowledge-distillation --client-dirs './sample_data/c1,./sample_data/c2' --generic-dir './sample_data/generic'
```
