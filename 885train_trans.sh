export PYTHONPATH=/mnt/hdd1/zhanghaonan/code/code_sgg_lxy/lib/cocoapi:/mnt/hdd1/zhanghaonan/code/code_sgg_lxy/Scene-Graph-Benchmark.pytorch-master/:$PYTHONPATH

export CUDA_VISIBLE_DEVICES=9
export NUM_GUP=2
MODEL_NAME="Motif_MCL"
NOWTIME=$(date "+%Y-%m-%d-%H-%M-%S")
echo "Training "${mode}" Nowtime is: "${NOWTIME}

if [ -d "./checkpoints/"${MODEL_NAME} ]; then
    echo "Current "${MODEL_NAME}" is exist, please change the checkpoint name."
    exit 0
fi

mkdir ./checkpoints/${MODEL_NAME}/
mkdir ./checkpoints/${MODEL_NAME}/code

cp ./tools/relation_train_net.py ./checkpoints/${MODEL_NAME}/code
cp ./maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py ./checkpoints/${MODEL_NAME}/code
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py ./checkpoints/${MODEL_NAME}/code
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py ./checkpoints/${MODEL_NAME}/code
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/model_prototype.py ./checkpoints/${MODEL_NAME}/code
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/relation_head.py ./checkpoints/${MODEL_NAME}/code
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/utils_prototype.py ./checkpoints/${MODEL_NAME}/code
cp ./scripts/885train_trans.sh ./checkpoints/${MODEL_NAME}/code
cp ./$0 ./checkpoints/${MOsDEL_NAME}/code

python  tools/relation_train_net.py \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
      MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
      MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
      MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor_sub_prototype_Memory \
      MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
      MODEL.ROI_RELATION_HEAD.CLASS_BALANCE_LOSS True \
      DTYPE "float32" \
      MODEL.ROI_HEADS.DETECTIONS_PER_IMG 80 \
      MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 1024 \
      SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH $NUM_GUP \
      SOLVER.BASE_LR 1e-3 SOLVER.PRE_VAL False \
      SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
      SOLVER.MAX_ITER 30000 SOLVER.VAL_PERIOD 130000 \
      SOLVER.STEPS "(28000, 48000)" \
      SOLVER.GRAD_NORM_CLIP 5.0 \
      SOLVER.CHECKPOINT_PERIOD 60000 GLOVE_DIR ./datasets/vg/ \
      MODEL.PRETRAINED_DETECTOR_CKPT /mnt/hdd1/zhanghaonan/code/code_sgg_lxy/Scene-Graph-Benchmark.pytorch-master/checkpoints/pretrained_faster_rcnn//model_final.pth \
      OUTPUT_DIR ./checkpoints/${MODEL_NAME};


