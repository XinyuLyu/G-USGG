# Introduction

## Training

### Detector

### Relation

```bash
CUDA_VISIBLE_DEVICES=<gpu_id> python -m torch.distributed.launch   --master_port <port> --nproc_per_node=2 tools/relation_train_net.py \
--config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" <args>
```

- master_port: 主节点的端口号
- nproc_per_node: 每个节点上的进程数量
- local_rank: 一台机器（一个节点上）进程的相对序号，节点之间的 local_rank 是独立的
- \<gpu_id\>: 指定在哪几张卡上进行分布式, e.g. 0,1,2,....
- \<port\>: 主节点端口号
- \<args\>: 其他参数，可追加

- args 可添加的参数
    + Training PredCls Task:
        ```bash
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODELROI_RELATION_HEADUSE_GT_OBJECT_LABEL True
        ```
    + Training SGCls Task:
        ```bash
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
        ```
    + Training SGDet Task:
        ```bash
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
        ```
    + Using different SGG models (choose one of below):
        ```bash
        # Motifs
        MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor
        # IMP (Iterative-Message-Passing)
        MODEL.ROI_RELATION_HEAD.PREDICTOR IMPPredictor
        # VCTree
        MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor
        # Transformer
        # 请注意，如果你使用 Transformer Model 需要修改 SOLVER.BASE_LR=0.001, SOLVER.SCHEDULE.TYPE=WarmupMultiStepLR, SOLVER.MAX_ITER=16000, SOLVER.IMS_PER_BATCH=16, SOLVER.STEPS=(10000, 16000) 参数
        MODEL.ROI_RELATION_HEAD.PREDICTOR TransformerPredictor
        # Unbiased-Causal-TDE
        MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor
        ```

    + Using Customize models:
        1. 参考 `maskrcnn-benchmark/modeling/roi_heads/relation_head/model_XXXXX.py` 和 `maskrcnn-benchmark/modeling/roi_heads/relation_head/utils_XXXXX.py` 创建这两个文件。
        2. 在 `maskrcnn-benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py` 添加对应的 `nn.Module`。
        3. 有时候你需要在 `maskrcnn-benchmark/modeling/roi_heads/relation_head/relation_head.py` 修改输入输出

## Evaluation

### Relation

与 Training 类似，使用 `relation_test_net.py` 文件并且需要添加 Test 相关的设置。
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/motif-precls-exmp \
OUTPUT_DIR /home/kaihua/checkpoints/motif-precls-exmp
```

### SGDet

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10027 --nproc_per_node=1 tools/relation_test_net.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False MODEL.ROI_RELATION_HEAD.PREDICTOR CausalAnalysisPredictor MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE TDE MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE sum MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER motifs TEST.IMS_PER_BATCH 1 DTYPE "float16" \
GLOVE_DIR /home/kaihua/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua/checkpoints/causal-motifs-sgdet \
OUTPUT_DIR ./checkpoints/causal-motifs-sgdet TEST.CUSTUM_EVAL True \
TEST.CUSTUM_PATH ./checkpoints/custom_images DETECTED_SGG_DIR ./checkpoints/your_output_path
```

## Some Examples

```bash
# pretrain detector
## training
### 4 gpus distributed
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10001 --nproc_per_node=4 tools/detector_pretrain_net.py --config-file "configs/e2e_relation_detector_R_101_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_ITER 40000 SOLVER.STEPS "(20000, 27000)" SOLVER.VAL_PERIOD 2000 SOLVER.CHECKPOINT_PERIOD 2000 MODEL.RELATION_ON False OUTPUT_DIR /home/kaihua.tkh/projects/benchmark_debug SOLVER.PRE_VAL False SOLVER.UPDATE_SCHEDULE_DURING_LOAD True

### 2 gpus distributed
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10002 --nproc_per_node=2 tools/detector_pretrain_net.py --config-file "configs/e2e_relation_detector_R_101_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 80000 SOLVER.STEPS "(40000, 57000)" SOLVER.VAL_PERIOD 5000 SOLVER.CHECKPOINT_PERIOD 5000 MODEL.RELATION_ON False OUTPUT_DIR /home/kaihua.tkh/projects/benchmark_debug SOLVER.PRE_VAL False SOLVER.UPDATE_SCHEDULE_DURING_LOAD True

### 1 gpu
CUDA_VISIBLE_DEVICES=0 python tools/detector_pretrain_net.py --config-file "configs/e2e_relation_detector_R_101_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 160000 SOLVER.STEPS "(80000, 108000)" SOLVER.VAL_PERIOD 5000 SOLVER.CHECKPOINT_PERIOD 5000 MODEL.RELATION_ON False OUTPUT_DIR /home/kaihua.tkh/projects/benchmark_debug SOLVER.UPDATE_SCHEDULE_DURING_LOAD True


## testing
### 4 gpus distributed
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10003 --nproc_per_node=4 tools/detector_pretest_net.py --config-file "configs/e2e_relation_detector_R_101_FPN_1x.yaml" DTYPE "float16" OUTPUT_DIR /home/kaihua.tkh/projects/benchmark_debug MODEL.RELATION_ON False



# relation prediction
## training
### 4 gpus distributed
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10004 --nproc_per_node=4 tools/relation_train_net.py --config-file "configs/e2e_relation_R_101_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_ITER 25000 SOLVER.STEPS "(8000, 16000)" GLOVE_DIR /home/kaihua.tkh/projects/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua.tkh/projects/model_final.pth OUTPUT_DIR /home/kaihua.tkh/checkpoints/relation_motif SOLVER.PRE_VAL False
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 10004 --nproc_per_node=4 tools/relation_train_net.py --config-file "configs/e2e_relation_R_101_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 16 TEST.IMS_PER_BATCH 4 DTYPE "float16" SOLVER.MAX_ITER 24000 SOLVER.STEPS "(12000, 16000)" GLOVE_DIR /data/share/glove.6B MODEL.PRETRAINED_DETECTOR_CKPT /data/sjx/exp/GQA/pretrain_detector/model_final.pth OUTPUT_DIR /data/sjx/exp/GQA/relation_motif SOLVER.PRE_VAL False


### 2 gpus distributed
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10005 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_R_101_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_ITER 80000 SOLVER.STEPS "(40000, 54000)" GLOVE_DIR /home/kaihua.tkh/projects/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua.tkh/projects/model_final.pth OUTPUT_DIR /home/kaihua.tkh/checkpoints/relation_motif SOLVER.PRE_VAL False

### 1 gpu
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/e2e_relation_R_101_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 3 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 120000 SOLVER.STEPS "(60000, 81000)" GLOVE_DIR /home/kaihua.tkh/projects/glove MODEL.PRETRAINED_DETECTOR_CKPT /home/kaihua.tkh/projects/model_final.pth OUTPUT_DIR /home/kaihua.tkh/checkpoints/relation_motif SOLVER.PRE_VAL False
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --config-file "configs/e2e_relation_R_101_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_ITER 90000 SOLVER.STEPS "(45000, 60000)" GLOVE_DIR /data/share/glove.6B MODEL.PRETRAINED_DETECTOR_CKPT /data/sjx/exp/GQA/pretrain_detector/model_final.pth OUTPUT_DIR /data/sjx/exp/GQA/relation_motif SOLVER.PRE_VAL False

## need to check the performance of model without frequence_bias
## added MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS = False for debug
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --master_port 10002 --nproc_per_node=2 tools/relation_train_net.py --config-file "configs/e2e_relation_R_101_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True SOLVER.IMS_PER_BATCH 8  DTYPE "float16" SOLVER.MAX_ITER 40000 SOLVER.STEPS "(20000, 27000)" SOLVER.VAL_PERIOD 5000 SOLVER.CHECKPOINT_PERIOD 5000 GLOVE_DIR /data4/jiaxin/glove MODEL.PRETRAINED_DETECTOR_CKPT /data4/jiaxin/benchmark_debug/model_final.pth OUTPUT_DIR /data4/jiaxin/relation_debug

# testing
CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py --config-file "configs/e2e_relation_R_101_FPN_1x.yaml" TEST.IMS_PER_BATCH 1 DTYPE "float16" GLOVE_DIR /home/kaihua/projects/glove SOLVER.PRE_VAL False OUTPUT_DIR /home/kaihua/checkpoints/TYPE_PATH_HERE

# image retrieval
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10005 --nproc_per_node=2 tools/image_retrieval_main.py --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" SOLVER.IMS_PER_BATCH 12 SOLVER.MAX_ITER 61 SOLVER.STEPS "(20, 50)" SOLVER.VAL_PERIOD 5 SOLVER.CHECKPOINT_PERIOD 5 SOLVER.SCHEDULE.TYPE "WarmupMultiStepLR" SOLVER.WARMUP_FACTOR 1.0 SOLVER.BASE_LR 0.01 SOLVER.PRE_VAL False OUTPUT_DIR /home/kaihua/checkpoints/TYPE_PATH_HERE 
```