python main.py --test --arch "srcnn" --test_model "CRNN" --resume "experiments/srcnn/model_best.pth"
python main.py --test --arch "srres" --test_model "CRNN" --resume "experiments/srres/model_best.pth"
python main.py --test --arch "tsrn"  --test_model "CRNN" --resume "experiments/tsrn/model_best.pth"
python main.py --test --arch "tg"    --test_model "CRNN" --resume "experiments/tg/model_best.pth"
python main.py --test --arch "tpgsr" --test_model "CRNN" --resume "experiments/tpgsr/model_best.pth"
python main.py --test --arch "tbsrn" --test_model "CRNN" --resume "experiments/tbsrn/model_best.pth"
python main.py --test --arch "tatt"  --test_model "CRNN" --resume "experiments/tbsrn/model_best.pth"

# 'ASTER', "CRNN", "MORAN",'ABINet','MATRN','PARSeq'