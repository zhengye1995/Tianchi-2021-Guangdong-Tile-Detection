'''
server中使用GPU做infer
面向工业瓷砖大赛demo
'''
from tools.inferServer import inferServer
import json
from mmdet.apis import init_detector_template, inference_detector_template, async_inference_detector
import cv2
import numpy as np

# RET = {
#     "name": "226_46_t20201125133518273_CAM1.jpg",
#     "image_height": 6000,
#     "image_width": 8192,
#     "category": 4,
#     "bbox": [
#         1587,
#         4900,
#         1594,
#         4909
#     ],
#     "score": 0.130577
# }


class myserver(inferServer):
    def __init__(self, model):
        super().__init__(model)
        print("init_myserver")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = device
        # self.model = model.to(device)
        self.model = model

    def pre_process(self, request):
        # print("my_pre_process.")
        # json process
        # file example
        file = request.files['img']
        file_t = request.files['img_t']
        # print(file.filename)
        file_data = file.read()
        file_data_t = file_t.read()
        img = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        img_t = cv2.imdecode(np.frombuffer(file_data_t, np.uint8), cv2.IMREAD_COLOR)
        return [img, img_t, file.filename]

    # pridict default run as follow：
    def pridect(self, data):
        img, img_t, filename = data
        result = inference_detector_template(self.model, img, img_t)
        return [result, filename]

    def post_process(self, result):
        data, filename = result
        if isinstance(data, tuple):
            bbox_result, _ = data
        else:
            bbox_result = data
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        predict_result = []
        max_score = 0

        for bbox, label in zip(bboxes, labels):
            xmin, ymin, xmax, ymax, score = bbox
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            dict_instance = dict()
            dict_instance['name'] = str(filename)
            dict_instance['category'] = int(label) + 1
            score = round(float(score), 6)
            dict_instance["score"] = score
            dict_instance["bbox"] = [xmin, ymin, xmax, ymax]
            predict_result.append(dict_instance)
            max_score = max(score, max_score)
        if max_score < 0.3:
            predict_result = []
        return json.dumps(predict_result)


if __name__ == '__main__':

    config_file = 'configs/experiment/round2/cas_dcn_r50_temp.py'
    checkpoint_file = 'work_dirs/cas_dcn_r50_temp/epoch_36.pth'

    # build the model from a config file and a checkpoint file
    mymodel = init_detector_template(config_file, checkpoint_file, device='cuda:0')
    myserver = myserver(mymodel)
    myserver.run(debuge=False)  # myserver.run("127.0.0.1", 1234)