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
![这是训练后的推理结果图](https://github.com/EMM-art-ark/FOCUS-WinterCamp-2026/blob/main/docs/test1.jpg)
