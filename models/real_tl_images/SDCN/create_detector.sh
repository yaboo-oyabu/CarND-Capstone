#!/bin/bash

ROOT_DIR="$(cd "$(dirname "$0")"; pwd -P)"

# Install object_detection_api
export PYTHONPATH=$PYTHONPATH:${ROOT_DIR}/models/research
export PYTHONPATH=$PYTHONPATH:${ROOT_DIR}/models/research/slim
python3 ${ROOT_DIR}/models/research/object_detection/builders/model_builder_test.py

# Download pretrained model
MODEL_NAME="ssd_inception_v2_coco_2017_11_17"
if [ ! -e ${ROOT_DIR}/pretrained_models/${MODEL_NAME} ]; then
    wget -P ${ROOT_DIR} http://download.tensorflow.org/models/object_detection/${MODEL_NAME}.tar.gz \
    && mkdir -p ${ROOT_DIR}/pretrained_models/ \
    && tar -zxf ${ROOT_DIR}/${MODEL_NAME}.tar.gz -C ${ROOT_DIR}/ \
    && rm ${ROOT_DIR}/${MODEL_NAME}.tar.gz \
    && mv ${ROOT_DIR}/${MODEL_NAME} ${ROOT_DIR}/pretrained_models/
fi

# Start training
python3 ${ROOT_DIR}/models/research/object_detection/train.py \
    --logtostderr \
    --train_dir=${ROOT_DIR}/detector_training \
    --pipeline_config_path=${ROOT_DIR}/pipeline_config/${MODEL_NAME}.config

# # Evaluation
# python3 ${ROOT_DIR}/models/research/object_detection/eval.py \
#     --logtostderr \
#     --checkpoint_dir ${ROOT_DIR}/detector_training \
#     --eval_dir=${ROOT_DIR}/detector_evaluation \
#     --pipeline_config_path ${ROOT_DIR}/pipeline_config/${MODEL_NAME}.config

# # https://github.com/tensorflow/tensorflow/issues/16268
# python3 ${ROOT_DIR}/models/research/object_detection/export_inference_graph.py \
#     --input_type image_tensor \
#     --pipeline_config_path ${ROOT_DIR}/pipeline_config/${MODEL_NAME}.config \
#     --trained_checkpoint_prefix ${ROOT_DIR}/detector_training/model.ckpt-2806 \
#     --output_directory ${ROOT_DIR}/ssd_inception_v2_retrained_2806
