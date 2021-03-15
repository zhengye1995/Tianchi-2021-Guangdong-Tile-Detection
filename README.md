
# 2021广东工业智造创新大赛 瓷砖瑕疵检测 解决方案

队伍：**我好菜**

## 比赛地址：[2021广东工业智造创新大赛 瓷砖瑕疵检测](https://tianchi.aliyun.com/competition/entrance/531846/information)

## Siamese Attention FPN 特征差+spatial attention思路
## core slides:
![Aaron Swartz](https://github.com/zhengye1995/Tianchi-2021-Guangdong-Tile-Detection/blob/main/temp_image/saf.png)
+ Siamese Attention FPN
    - 基本框架：Cascade-RCNN
    - backbone： resnet50
    - neck: FPN 基础上引入特征差和空间attention
    - RPN: 
        - anchor box assign 采用 ATSS
        - anchor_scale 随训练尺度一起缩小
    - cascade 三个head 根据比赛map计算iou进行对应调整
    - 采用fp16加速训练和增大输入面积来缓解部分小目标问题
+ 后处理
    - NMS
    - 最大score二类后处理, 根据单个图片bbox最高score来对图像进行二次过滤, 判断该图像是否是正常样本

## 代码环境及依赖 （阿里线上环境）

+ OS: Ubuntu
+ GPU: Telsa V100 (16GB) * 4 (实测训练只需要7-8g显存)
+ python: python3.7
+ nvidia 依赖:
   - cuda: 10或者11均可
   - cudnn: 7.5.1-8 均可
   - nvidia driver version: 430.14 - 最新均可
+ deeplearning 框架: pytorch1.6
+ 其他依赖请参考requirement.txt （和mmdetection一致）
   

## 模型训练及预测 （在比赛平台docker方式提交运行）
    
   ./run.sh
   
## Contact

    author：rill

    email：18813124313@163.com


