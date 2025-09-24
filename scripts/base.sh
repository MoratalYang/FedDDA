python federated_main.py --trainer FEDDDA --dataset Office31 --beta 0.0 --device_id 0 OPTIM.MAX_EPOCH 1 &
python federated_main.py --trainer FEDDDA --dataset OfficeHome --beta 0.0 --device_id 1 OPTIM.MAX_EPOCH 1 &
python federated_main.py --trainer FEDDDA --dataset pacs --beta 0.0 --device_id 2 OPTIM.MAX_EPOCH 1 &
python federated_main.py --trainer FEDDDA --dataset domainnet --beta 0.0 --device_id 3 OPTIM.MAX_EPOCH 1 &