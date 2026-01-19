# 测试命令汇总（MIBF‑Net / ConNeXT，HAM / Spine）

下面命令均为测试流程（实际命令更改路径，下面命令是我在自己文件夹下运行的路径），输出 CSV 预测文件。ConNeXT那个多了一个金标准路径，用于计算acc。

## MIBF（ResNet50）- HAM 92.91%

```bash
CUDA_VISIBLE_DEVICES=5 python predict_HAM.py \
  --image_dir /data/QLI/Assignment_for_dl/Data/data_for_dl/HAM10000_dataset/test \
  --json_path /data/QLI/Assignment_for_dl/Data/data_for_dl/HAM10000_dataset/response/test/responses.json \
  --model_path /data/QLI/Assignment_for_dl/lib_dl/best_HAM_92.91_MIBF.pth \
  --output_path /data/QLI/Assignment_for_dl/lib_dl/results/mibf_net/preds_ham_test.csv
```

## MIBF（ResNet50）- Spine 91.52%

```bash
CUDA_VISIBLE_DEVICES=5 python predict_Spine.py \
  --image_dir /data/QLI/Assignment_for_dl/Data/data_for_dl/Spine_dataset/test/images \
  --json_path /data/QLI/Assignment_for_dl/Data/data_for_dl/Spine_dataset/response/test/responses.json \
  --model_path /data/QLI/Assignment_for_dl/lib_dl/best_Spine_91_MIBF.pth \
  --output_path /data/QLI/Assignment_for_dl/lib_dl/results/mibf_net/preds_spine_test.csv
```

## ConNeXT - HAM  93.41%

```bash
cd /data/QLI/Assignment_for_dl/lib_dl
CUDA_VISIBLE_DEVICES=5 python predict_ConvNext.py \
  --image_dir /data/QLI/Assignment_for_dl/Data/data_for_dl/HAM10000_dataset/test \
  --json_path /data/QLI/Assignment_for_dl/Data/data_for_dl/HAM10000_dataset/response/test/responses.json \
  --model_path /data/QLI/Assignment_for_dl/runs/ConNexT/ConNexT_HAM10000_20260118_155533/checkpoints/epoch=171-acc_val_Accuracy=0.9341-f1_val_F1_macro=0.8926-auc_val_AUROC=0.9848.ckpt \
  --output_path  \
  --config /data/QLI/Assignment_for_dl/lib_dl/ConNexT/config_ham.yaml \
  --label_csv (这里填GT标签路径)
```



