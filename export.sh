ls -alrt checkpoint/*/*/*.h5 | awk '{print $9}'
echo
echo -n 'input model path for export : '
read MODEL_PATH

ONNX_SAVE_PATH="${MODEL_PATH:0:-3}.onnx"
INPUT_LAYER_NAME=sbd_input:0
OUTPUT_LAYER_NAMES=Identity:0,Identity_1:0,Identity_2:0,Identity_3:0,Identity_4:0
RENAME_INPUT_LAYER=sbd_input
RENAME_OUTPUT_LAYERS=sbd_output_0,sbd_output_1,sbd_output_2,sbd_output_3,sbd_output_4
python -m tf2onnx.convert \
    --keras $MODEL_PATH \
    --output $ONNX_SAVE_PATH \
    --inputs $INPUT_LAYER_NAME \
    --outputs $OUTPUT_LAYER_NAMES \
    --rename-inputs $RENAME_INPUT_LAYER \
    --rename-outputs $RENAME_OUTPUT_LAYERS \
    --inputs-as-nchw $INPUT_LAYER_NAME \
    --outputs-as-nchw $OUTPUT_LAYER_NAMES \
    --opset 11
