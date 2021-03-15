python tools/process_data/label2json.py

python tools/process_data/label2json2.py

./tools/dist_train.sh configs/experiment/round2/cas_dcn_r50_temp.py 4

python tools/inference_server_template.py