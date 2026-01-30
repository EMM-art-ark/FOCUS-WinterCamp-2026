# 每日调试心得

---

## 1.24
今天完成了基本调试环境的搭建，完成任务验收点一的相关要求。其中Hello World脚本是仓库src文件夹中的Hello World.sh文件，属于shell脚本，而自我介绍的GitHub Pages的网址在https://emm-art-ark.github.io/  
今天的主要工作是学习了github仓库的创建、SSH key的配置，学习了GitHub Pages的创建和编辑，安装了git bash来实现Git SSH密钥的添加。为了编写Hello World脚本，今天还了解了Shell相关的部分内容和语法。  
此外，今天还学习了有关python与pytorch深度学习的有关内容，特别是卷积神经网络（CNN）的相关算法和实现方法。第二天就可以开始训练模型来进行手写数字识别了。

## 1.25
今天完成了CNN手写数字识别的模型训练，准确率达到了96.37%。这次CNN手写数字识别的模型训练使用的是LeNet-5模型，利用MNIST数据集进行模型训练。  
代码主要分为以下几个方面：  
1. 引入模型训练所需模块，如torch、torchvision等
2. 获取所需MNIST数据集，通过torchvision.datasets.MNIST方法分别获取测试集和训练集，并使用transforms.Compose对数据进行预处理
3. 使用torch.utils.data.DataLoader对数据进行分批次处理
4. 定义模型CNN并实例化，通过torch.nn.Sequential按层顺序封装模型
5. 定义损失函数loss_fn = nn.CrossEntropyLoss()
6. 定义学习率和优化器optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
7. 定义训练轮数和损失量列表，便于后续训练模型和绘制图形
8. 训练模型
9. 绘制损失随训练轮数减少的图形
10. 最后训练一轮，计算最终模型准确率

关于本次模型训练所用的参数，数据分批次处理时batch_size=256，学习率learning_rate = 0.6，训练轮数epochs = 10  
![这是本模型损失随训练轮数减少的图形](https://github.com/EMM-art-ark/FOCUS-WinterCamp-2026/blob/main/docs/LeNet-5-2.png)  
![这是训练时的截图](https://github.com/EMM-art-ark/FOCUS-WinterCamp-2026/blob/main/docs/LeNet.png)

## 1.26
今天学习了YOLO的相关使用方法，并利用YOLO官方的coco128数据集训练了模型，使用训练得到的模型来完成对指定图片的推理验证。其中官方预训练模型是yolo26n.pt，训练时相关参数workers=0, epochs=100, batch=16, device=-1。  
以下是训练和验证时的相关图片  
![这是训练结果图片](https://github.com/EMM-art-ark/FOCUS-WinterCamp-2026/blob/main/docs/results.png)
![这是推理测试结果图片](https://github.com/EMM-art-ark/FOCUS-WinterCamp-2026/blob/main/docs/test1.jpg)

## 1.27
今天利用自己的数据集进行训练，并对训练后的模型进行推理测试。  
模型训练所用的数据集是自己拍照得到的50张照片，其中40张用作训练集，10张用作测试集。官方预训练模型是yolo26n.pt，训练过程中相关参数workers=0,epochs=500,batch=16,device=[-1,-1]。训练所用labels是自己设定的，所用的.yaml文件由自己编写。  
今天工作大致流程为：  
1. 准备训练所用数据集
2. 对数据集中的图片进行数据标注，以获得labels对应.txt文件
3. 整理数据集文件，将其存放在datasets/focus中，分为images和labels两个文件夹，每个文件夹又进一步细分为train和val两个子文件夹
4. 按照基本格式编写训练模型所用.yaml文件，命名为focus.yaml
5. 对模型进行训练
6. 对模型训练效果进行测试

以下是今日训练和验证时的相关图片  
![这是训练模型得到的图片](https://github.com/EMM-art-ark/FOCUS-WinterCamp-2026/blob/main/docs/train.png)  
根据程序运行提示，模型训练187轮后停止训练，因为在训练87轮之后，模型就已经没有可见的提升了，程序在继续100轮训练后自动停止训练
![这是预测图](https://github.com/EMM-art-ark/FOCUS-WinterCamp-2026/blob/main/docs/1.jpg)

## 1.28
今天由于一整天都在户外，没有时间进行模型训练和验证，之后两天会把精力聚焦在对模型的学习和后续工作上。

## 1.29
今天学习了后面选作部分的内容。由于是第一次学习，所以配置文件方面不是很顺利。在了解相关算法时，对人工智能和深度学习的应用有了更深入的了解。  

## 1.30
今天继续学习细粒度图像识别算法，对其基本方式有了了解，但代码方面并没有很多突破。尽管冬令营已经进入尾声，我还是愿意在今后的学习中加强对深度学习方面的认识，进一步提升自己。
